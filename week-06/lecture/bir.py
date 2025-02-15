import numpy as np
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from itertools import product

num_states = 2
num_actions = 2
states = [0, 1]
actions = [0, 1]


def mdp(state: int, action: int) -> (int, float):
    """
    Given a state and action, return (next_state, reward).
    The reward is sampled from a Normal distribution.
    """
    assert state in [0, 1], "Invalid state"
    assert action in [0, 1], "Invalid action"

    if state == 0:
        if action == 0:
            next_state = 0 if np.random.rand() < 0.9 else 1
            reward = np.random.normal(2.0, 1.0)
            return next_state, reward
        elif action == 1:
            next_state = 0 if np.random.rand() < 0.1 else 1
            reward = np.random.normal(1.5, 1.0)
            return next_state, reward
    elif state == 1:
        if action == 0:
            next_state = 0 if np.random.rand() < 0.9 else 1
            reward = np.random.normal(0.0, 1.0)
            return next_state, reward
        elif action == 1:
            next_state = 0 if np.random.rand() < 0.2 else 1
            reward = np.random.normal(3.0, 1.0)
            return next_state, reward
    assert False, "Should not reach here"


def q_model(state: int, action: int, reward: float, next_state: int, mu_prior, sigma_prior, gamma):
    """
    Bayesian Q-learning model that considers full history of transitions.
    """
    q_values = numpyro.sample("q_values", dist.Normal(mu_prior, sigma_prior))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
    current_q = q_values[state, action]

    next_q = jnp.max(q_values[next_state])
    td_target = reward + gamma * next_q

    numpyro.sample("obs",
                   dist.Normal(current_q, sigma),
                   obs=td_target)


def perform_inference(
        rng_key,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        mu_prior,
        sigma_prior,
        gamma=0.8
) -> MCMC:
    """
    Perform inference using historical data for better Q-value estimates.
    """
    kernel = NUTS(q_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, progress_bar=False)
    mcmc.run(rng_key,
             state=state,
             action=action,
             reward=reward,
             next_state=next_state,
             mu_prior=mu_prior,
             sigma_prior=sigma_prior,
             gamma=gamma
             )
    return mcmc


def get_temperature(t, initial_temp=3.0, min_temp=0.8, decay=0.001):
    return max(min_temp, initial_temp * np.exp(-decay * t))


def main():
    history = {(s, a): {'rewards': [], 'next_states': []}
               for s in states for a in actions}
    current_state = 1
    num_steps = 100
    rng_key = jax.random.PRNGKey(0)

    # Initialize with wider priors to encourage exploration
    mu_prior = jnp.zeros((num_states, num_actions))
    sigma_prior = 15. * jnp.ones((num_states, num_actions))
    mcmc = None

    print("Bayesian Q-Learning in a 2-state MDP (using Thompson Sampling)\n")
    print("Environment dynamics:")
    print("State 0:")
    print("  Action 0: reward ~ N(2.0, 1.0), P(stay) = 0.9")
    print("  Action 1: reward ~ N(1.5, 1.0), P(transition) = 0.9")
    print("State 1:")
    print("  Action 0: reward ~ N(0.0, 1.0), P(stay) = 0.9")
    print("  Action 1: reward ~ N(3.0, 1.0), P(transition) = 0.9\n")

    for t in range(num_steps):
        # Thompson sampling with temperature scaling
        qs = []

        for a in actions:
            if len(history[(current_state, a)]['rewards']) == 0 or mcmc is None:
                q_choice = np.random.normal(0, 10.0)
            else:
                q_samples = mcmc.get_samples()["q_values"][:, current_state, a]
                q_choice = np.random.choice(q_samples)
            qs.append(q_choice)

        chosen_action = int(np.argmax(qs))
        next_state, reward = mdp(current_state, chosen_action)

        # Store transition for logging
        history[(current_state, chosen_action)]['rewards'].append(reward)
        history[(current_state, chosen_action)]['next_states'].append(next_state)

        # Update posterior distributions
        rng_key, subkey = jax.random.split(rng_key)
        mcmc = perform_inference(
            subkey,
            current_state,
            chosen_action,
            reward,
            next_state,
            mu_prior,
            sigma_prior
        )
        q_est = mcmc.get_samples()["q_values"]
        mu_s = jnp.mean(q_est, axis=0)
        sigma_s = jnp.std(q_est, axis=0)

        alpha = 0.1
        mu_prior = (1 - alpha) * mu_prior + alpha * jnp.array(mu_s)
        sigma_prior = (1 - alpha) * sigma_prior + alpha * jnp.array(sigma_s)

        current_state = next_state

        # Print log
        print("=========================================")
        print(f"Step {t + 1}: Chose action {chosen_action} in state {current_state}")
        print(f"  Estimated Q-values: {qs}")
        print(f"  Reward: {reward}\n")
        print(f"  Visits: ")

        for s, a in product(states, actions):
            print(f"    Q({s}, {a}): {len(history[(s, a)]['rewards'])}")

        if t % 10 == 0:
            print("=========================================")
            print("\nQ-value estimates:")
            q_estimates = {}
            for s, a in product(states, actions):
                if len(history[(s, a)]['rewards']) > 0:
                    rng_key, subkey = jax.random.split(rng_key)
                    q_est = mcmc.get_samples()["q_values"][:, s, a]
                    mean = np.mean(q_est)
                    std = np.std(q_est)
                    q_estimates[(s, a)] = (mean, std)
                    print(f"  Q({s}, {a}) = {mean:.2f} ± {std:.2f}")

    # Print final Q-value estimates
    print("\nFinal Q-value estimates:")
    final_estimates = {}
    for s, a in product(states, actions):
        if len(history[(s, a)]['rewards']) > 0:
            rng_key, subkey = jax.random.split(rng_key)
            q_est = mcmc.get_samples()["q_values"][:, s, a]
            mean = np.mean(q_est)
            std = np.std(q_est)
            final_estimates[(s, a)] = (mean, std)
            print(f"  Q({s}, {a}) = {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    main()
