import stormpy
import stormpy.simulator
import random

from typing import Any, Optional


def print_state_info(state_id, _observation, labels, step):
    print(f"\n--- Step {step} ---")
    print(f"State ID: {state_id}")

    # Get variable values from state valuations
    if model.has_state_valuations():
        valuation = model.state_valuations.get_state(state_id)
        var_values = {}
        for var in valuation:
            var_values[var.name] = var.get_value()
        print(
            f"State variables: weather={var_values.get('weather', '?')}, mood={var_values.get('mood', '?')}, status={var_values.get('status', '?')}"
        )

    # Get state labels (from simulator result)
    filtered_labels = [label for label in labels if label != "init"]
    if filtered_labels:
        print(f"State labels: {', '.join(filtered_labels)}")


def simulate_random_policy(
    model: Any,
    reward_model: Optional[Any],
    num_episodes: int = 1,
    max_steps: int = 10,
    seed: int = 0,
) -> None:
    for episode in range(num_episodes):
        print(f"=== Episode {episode + 1} ===")

        # Create simulator
        sim_seed = seed + episode
        simulator = stormpy.simulator.create_simulator(model, seed=sim_seed)
        state = simulator.restart()

        total_reward = 0
        step = 0

        print_state_info(state[0], state[1], state[2], step)

        while not simulator.is_done() and step < max_steps:
            available_actions = simulator.available_actions()
            assert len(available_actions) > 0, "No available actions (terminal state)"

            # Get current state ID before taking action
            current_state_id = state[0]

            # Randomly choose an action and take a step
            chosen_action = random.choice(available_actions)
            state = simulator.step(chosen_action)
            state_id, observation, labels = state

            # Get reward for the state-action pair
            reward_value = 0
            if reward_model and reward_model.has_state_action_rewards:
                # Calculate global choice index: row_group_start + action_offset
                row_group_start = model.transition_matrix.get_row_group_start(
                    current_state_id
                )
                choice_idx = row_group_start + chosen_action
                reward_value = reward_model.state_action_rewards[choice_idx]
            elif reward_model and reward_model.has_state_rewards:
                reward_value = reward_model.state_rewards[state_id]

            if reward_value > 0:
                print(f"*** Received reward: {reward_value} ***")
            total_reward += reward_value

            step += 1
            print_state_info(state_id, observation, labels, step)

        print()
        print(f"Episode finished: Total reward = {total_reward}")


if __name__ == "__main__":
    random.seed(42)
    prism_program = stormpy.parse_prism_program("activity_agent.pm")
    model = stormpy.build_model(prism_program)

    # Get the reward model
    reward_model = None
    if len(model.reward_models) > 0:
        reward_model_name = list(model.reward_models.keys())[0]
        reward_model = model.reward_models[reward_model_name]
        print(f"Reward model: '{reward_model_name}'")

    simulate_random_policy(model, reward_model, num_episodes=1, seed=42)
