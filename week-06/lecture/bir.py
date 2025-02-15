import numpy as np
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from itertools import product

# Environment setup
num_states = 2
num_actions = 2
states = [0, 1]
actions = [0, 1]


def step(state: int, action: int) -> (int, float):
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


def q_model(
        state: int,
        action: int,
        rewards,
        next_states,
        mu_prior,
        sigma_prior,
        gamma):
    """
    Bayesian Q-learning model that considers full history of transitions.
    """
    # Sample Q-values from prior
    q_values = numpyro.sample("q_values", dist.Normal(mu_prior, sigma_prior))

    # Sample observation noise
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))

    # Current Q-value for this state-action pair
    current_q = q_values[state, action]

    # For each observed transition
    with numpyro.plate("data", len(rewards)):
        # Calculate TD targets using next state maximum Q-values
        next_q = jnp.max(q_values[next_states], axis=-1)
        td_target = rewards + gamma * next_q

        # Observe TD targets with noise
        numpyro.sample("obs",
                       dist.Normal(current_q, sigma),
                       obs=td_target)


def perform_inference(
        rng_key,
        state: int,
        action: int,
        rewards: list,
        next_states: list,
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
             rewards=jnp.array(rewards),
             next_states=jnp.array(next_states),
             mu_prior=mu_prior,
             sigma_prior=sigma_prior,
             gamma=gamma
             )
    return mcmc


def get_temperature(t, initial_temp=3.0, min_temp=0.8, decay=0.001):
    return max(min_temp, initial_temp * np.exp(-decay * t))


def main():
    # Initialize tracking dictionaries for historical data
    history = {(s, a): {'rewards': [], 'next_states': []}
               for s in states for a in actions}

    # Setup initial state and random key
    current_state = 0
    num_steps = 100
    rng_key = jax.random.PRNGKey(0)

    # Initialize with wider priors to encourage exploration
    mu_prior = jnp.zeros((num_states, num_actions))
    sigma_prior = 10.0 * jnp.ones((num_states, num_actions))

    print("Bayesian Q-Learning in a 2-state MDP (using Thompson Sampling)\n")
    print("Environment dynamics:")
    print("State 0:")
    print("  Action 0: reward ~ N(2.0, 1.0), P(stay) = 0.9")
    print("  Action 1: reward ~ N(1.5, 1.0), P(transition) = 0.9")
    print("State 1:")
    print("  Action 0: reward ~ N(0.0, 1.0), P(stay) = 0.9")
    print("  Action 1: reward ~ N(3.0, 1.0), P(transition) = 0.9\n")

    # Learning loop
    for t in range(num_steps):
        # Thompson sampling with temperature scaling
        qs = []
        temp = get_temperature(t)

        for a in actions:
            if len(history[(current_state, a)]['rewards']) == 0:
                # Use prior sampling for unexplored actions
                q_choice = np.random.normal(0, 10.0)
            else:
                # Use MCMC inference for explored actions
                rng_key, subkey = jax.random.split(rng_key)
                mcmc = perform_inference(
                    subkey,
                    current_state,
                    a,
                    history[(current_state, a)]['rewards'],
                    history[(current_state, a)]['next_states'],
                    mu_prior,
                    sigma_prior
                )
                q_samples = mcmc.get_samples()["q_values"][:, current_state, a]
                q_choice = np.random.choice(q_samples) / temp
            qs.append(q_choice)

        chosen_action = int(np.argmax(qs))

        # Take action and observe
        next_state, reward = step(current_state, chosen_action)

        # Store transition in history
        history[(current_state, chosen_action)]['rewards'].append(reward)
        history[(current_state, chosen_action)]['next_states'].append(next_state)

        # Update priors periodically
        if t % 5 == 0:
            _mu_priors = []
            _sigma_priors = []
            for s, a in product(states, actions):
                if len(history[(s, a)]['rewards']) > 0:
                    rng_key, subkey = jax.random.split(rng_key)
                    mcmc = perform_inference(
                        subkey,
                        s,
                        a,
                        history[(s, a)]['rewards'],
                        history[(s, a)]['next_states'],
                        mu_prior,
                        sigma_prior
                    )
                    q_est = mcmc.get_samples()["q_values"][:, s, a]
                    mu = np.mean(q_est)
                    sigma = np.std(q_est)
                else:
                    mu = 0.0
                    sigma = 15.0
                _mu_priors.append(mu)
                _sigma_priors.append(sigma)

            # Smooth update of priors
            alpha = 0.1  # Learning rate for prior updates
            mu_prior = (1 - alpha) * mu_prior + alpha * jnp.array(_mu_priors).reshape(num_states, num_actions)
            sigma_prior = (1 - alpha) * sigma_prior + alpha * jnp.array(_sigma_priors).reshape(num_states, num_actions)

        current_state = next_state

        print("=========================================")
        print(f"Step {t + 1}: Chose action {chosen_action} in state {current_state}")
        print(f"  Temperature: {temp:.2f}")
        print(f"  Estimated Q-values: {qs}")
        print(f"  Reward: {reward}\n")
        print(f"  mu_prior:\n{mu_prior}")
        print(f"  sigma_prior:\n{sigma_prior}")
        print(f"  Visits: ")
        for s, a in product(states, actions):
            print(f"    Q({s}, {a}): {len(history[(s, a)]['rewards'])}")

    # Print final Q-value estimates
    print("\nFinal Q-value estimates:")
    final_estimates = {}
    for s, a in product(states, actions):
        if len(history[(s, a)]['rewards']) > 0:
            rng_key, subkey = jax.random.split(rng_key)
            mcmc = perform_inference(
                subkey,
                s,
                a,
                history[(s, a)]['rewards'],
                history[(s, a)]['next_states'],
                mu_prior,
                sigma_prior
            )
            q_est = mcmc.get_samples()["q_values"][:, s, a]
            mean = np.mean(q_est)
            std = np.std(q_est)
            final_estimates[(s, a)] = (mean, std)
            print(f"  Q({s}, {a}) = {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    main()