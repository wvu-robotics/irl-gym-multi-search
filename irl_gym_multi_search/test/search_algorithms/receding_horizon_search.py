# import numpy as np
# from irl_gym_multi_search.test.update_distribution import update_distribution

# # Define possible actions
# actions = {
#     0: np.array([0, -1]),  # Up
#     1: np.array([-1, 0]),  # Left
#     2: np.array([0, 1]),   # Down
#     3: np.array([1, 0])    # Right
# }

# def simulate_step(size_x, size_y, position, action, obs, fov_dict, obstacles, temp_distribution, boundary_penalty=-0.1):
#     """
#     Simulates a single step of the agent in the environment.
    
#     Args:
#     - size_x, size_y: Dimensions of the environment.
#     - position: Current position of the agent.
#     - action: Action to be taken by the agent.
#     - obs: Current observation.
#     - fov_dict: Field of view dictionary.
#     - obstacles: List of obstacles in the environment.
#     - temp_distribution: Temporary distribution for the agent.
#     - boundary_penalty: Penalty for attempting to move outside the environment boundaries.
    
#     Returns:
#     - new_position: New position of the agent as a 2D list.
#     - temp_distribution: Updated distribution after taking the step.
#     - step_penalty: Penalty incurred in this step (0 if no boundary violation, boundary_penalty otherwise).
#     """
#     if action not in actions:
#         raise ValueError(f"Invalid action: {action}")

#     new_position = np.array(position) + actions[action]
#     new_position = new_position.reshape(1, -1)

#     # Check for boundary violations and apply penalty if necessary
#     off_grid = np.any(new_position < 0) or new_position[0, 0] >= size_x or new_position[0, 1] >= size_y
#     step_penalty = boundary_penalty if off_grid else 0  # Apply penalty if moving outside the grid

#     new_position = np.clip(new_position, [0, 0], [size_x - 1, size_y - 1])

#     observation = {0: np.array([[obs[0][0]]])}
#     current_fov = {0: np.array([[1]])}
#     new_position_list = new_position.tolist()
#     temp_distribution = update_distribution(size_x, size_y, 1, new_position_list, current_fov, observation, fov_dict, obstacles, temp_distribution)

#     return new_position_list, temp_distribution, step_penalty


# def receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, fov_dict, obstacles, last_action=None, horizon=5, rollout=5, epsilon=0.1, boundary_penalty=-0.1):
#     best_action = None
#     max_reward = -np.inf

#     for initial_action in range(4):  # Iterate over all possible initial actions.
#         total_reward = 0  # Initialize total reward for this action path

#         for _ in range(rollout):  # Perform rollouts to estimate the value of each action
#             position = cur_pos
#             temp_distribution_path = temp_distribution.copy()
#             for h in range(horizon):  # Simulate steps within the horizon
#                 if np.random.rand() < epsilon:  # Exploration
#                     action = np.random.randint(4)
#                 else:  # Exploitation
#                     action_rewards = []
#                     action_rewards_rounded = []
#                     for action_option in range(4):  # Evaluate all possible actions
#                         predicted_position, _, step_penalty = simulate_step(size_x, size_y, position, action_option, obs, fov_dict, obstacles, temp_distribution_path, boundary_penalty)
#                         # Make sure to properly access the reward for the predicted position
#                         reward = temp_distribution_path[predicted_position[0][0]][predicted_position[0][1]] + step_penalty
#                         action_rewards.append(reward)
#                         action_rewards_rounded.append(np.round(reward, 8))
#                     print('Action rewards: ', action_rewards_rounded)
#                     action = np.argmax(action_rewards)  # Choose the action with the highest reward
#                     print('Action selected: ', action)

#                 # Simulate the step with the chosen action
#                 position, temp_distribution_path, step_penalty = simulate_step(size_x, size_y, position, action, obs, fov_dict, obstacles, temp_distribution_path, boundary_penalty)
#                 print(position[0][0], position[0][1])
#                 # print(temp_distribution_path[0][0])
#                 # Calculate and accumulate rewards, ensuring future rewards are considered
#                 reward_at_position = temp_distribution_path[position[0][0]][position[0][1]]
#                 print('Reward at position: ', reward_at_position)
#                 print(temp_distribution_path)
#                 total_reward += (reward_at_position + step_penalty) * (0.95 ** h)
#         # avg_reward = total_reward / (rollout * horizon)
#         avg_reward = total_reward / rollout

#         if avg_reward > max_reward:
#             max_reward = avg_reward
#             best_action = initial_action

#     print(f"Final Best Action: {best_action}, Max Reward: {max_reward}")
#     new_position, temp_distribution, _ = simulate_step(size_x, size_y, cur_pos, best_action, obs, fov_dict, obstacles, temp_distribution, boundary_penalty)
#     return [best_action], temp_distribution, max_reward



import numpy as np
import copy
from irl_gym_multi_search.test.update_distribution import update_distribution

# Define possible actions
actions = {
    0: np.array([0, -1]),  # Up
    1: np.array([-1, 0]),  # Left
    2: np.array([0, 1]),   # Down
    3: np.array([1, 0])    # Right
}

def simulate_step(size_x, size_y, position, action, obs, fov_dict, obstacles, temp_distribution, boundary_penalty=-0.1):
    # Calculate new position based on the action
    new_position = np.array(position) + actions[action]
    new_position = new_position.reshape(1, -1)

    # Check for boundary violations and apply penalty if necessary
    step_penalty = 0
    if np.any(new_position < 0) or new_position[0, 0] >= size_x or new_position[0, 1] >= size_y:
        step_penalty = boundary_penalty
        # print('boundary penalty with action: ', action)

    # Ensure new position does not exceed boundaries
    new_position = np.clip(new_position, [0, 0], [size_x - 1, size_y - 1])

    current_fov = {0: np.array([[1]])}
    observation = {0: np.array([[obs[0][0]]])}

    # Simulate observation for new position
    # Assuming update_distribution function updates temp_distribution based on new_position and observation
    updated_distribution = update_distribution(size_x, size_y, 1, new_position, current_fov, observation, fov_dict, obstacles, temp_distribution)

    return new_position.tolist(), updated_distribution, step_penalty

import numpy as np
import copy  # For deepcopy

def receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, fov_dict, obstacles, last_action=None, horizon=5, rollout=5, epsilon=0.1, boundary_penalty=-10.0):
    best_path = []
    max_path_reward = -np.inf  # Initialize with a very low reward

    # Rollout loop
    for _ in range(rollout):
        temp_distribution_copy = copy.deepcopy(temp_distribution)  # Deepcopy of the distribution for this rollout
        current_position = cur_pos[:]  # Copy current position to modify during the rollout
        path_reward = 0  # Initialize reward for this path
        path_actions = []  # Initialize list to store actions for this path

        # Horizon loop
        for _ in range(horizon):
            pre_observation_distribution = copy.deepcopy(temp_distribution_copy)  # Copy the distribution before observation

            if np.random.rand() < epsilon:  # Exploration: choose a random action
                action = np.random.randint(0, 4)
                current_position, temp_distribution_copy, step_penalty = simulate_step(size_x, size_y, current_position, action, obs, fov_dict, obstacles, temp_distribution_copy, boundary_penalty)
                reward = pre_observation_distribution[current_position[0][0]][current_position[0][1]] + step_penalty
            else:  # Exploitation: choose the best action based on reward
                action_rewards = []
                for action_option in range(4):  # Loop through all possible actions
                    current_position, simulated_distribution, step_penalty = simulate_step(size_x, size_y, current_position, action_option, obs, fov_dict, obstacles, temp_distribution_copy, boundary_penalty)
                    temp_reward = pre_observation_distribution[current_position[0][0]][current_position[0][1]] + step_penalty
                    action_rewards.append(temp_reward)
                # print(action_rewards)
                # Select the action with the highest reward
                action = np.argmax(action_rewards)
                reward = np.max(temp_reward)
            
            # print('Action: ', action, 'Reward: ', reward)
            path_actions.append(action)  # Append the selected action to the path
            path_reward += reward  # Accumulate rewards for this path

        # Compare the path's reward with the max reward and update if this path is better
        if path_reward > max_path_reward:
            max_path_reward = path_reward
            best_path = path_actions
        # print(best_path)

    # Select the first action of the best path
    best_action = best_path[0] if best_path else None

    # Return the first action of the best path, the distribution after the first action, and the max path reward
    new_position, updated_distribution, _ = simulate_step(size_x, size_y, cur_pos, best_action, obs, fov_dict, obstacles, temp_distribution, boundary_penalty)
    return [best_action], updated_distribution, max_path_reward


# def receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, fov_dict, obstacles, last_action=None, horizon=5, rollout=5, epsilon=0.1, boundary_penalty=-0.1):
#     best_path = []
#     max_path_reward = -np.inf  # Initialize with a very low reward

#     # Rollout loop
#     for _ in range(rollout):
#         temp_distribution_copy = copy.deepcopy(temp_distribution)  # Deepcopy of the distribution for this rollout
#         current_position = cur_pos[:]  # Copy current position to modify during the rollout
#         path_reward = 0  # Initialize reward for this path
#         path_actions = []  # Initialize list to store actions for this path

#         # Horizon loop
#         for _ in range(horizon):
#             if np.random.rand() < epsilon:  # Exploration: choose a random action
#                 action = np.random.randint(0, 4)
#                 current_position, temp_distribution_copy, step_penalty = simulate_step(size_x, size_y, current_position, action, obs, fov_dict, obstacles, temp_distribution_copy, boundary_penalty)
#                 reward = temp_distribution_copy[current_position[0][0]][current_position[0][1]] + step_penalty
#             else:  # Exploitation: choose the best action based on reward
#                 action_rewards = []
#                 for action_option in range(4):  # Loop through all possible actions
#                     current_position, simulated_distribution, step_penalty = simulate_step(size_x, size_y, current_position, action_option, obs, fov_dict, obstacles, temp_distribution_copy, boundary_penalty)
#                     temp_reward = simulated_distribution[current_position[0][0]][current_position[0][1]] + step_penalty
#                     action_rewards.append(temp_reward)
#                 print(action_rewards)
#                 # Select the action with the highest reward
#                 action = np.argmax(action_rewards)
#                 reward = temp_distribution_copy[current_position[0][0]][current_position[0][1]] + step_penalty

#             print('Action: ', action, 'Reward: ', reward)
#             path_actions.append(action)  # Append the selected action to the path
#             path_reward += reward  # Accumulate rewards for this path

#         # Compare the path's reward with the max reward and update if this path is better
#         if path_reward > max_path_reward:
#             max_path_reward = path_reward
#             best_path = path_actions

#     # Select the first action of the best path
#     best_action = best_path[0] if best_path else None

#     # Return the first action of the best path, the distribution after the first action, and the max path reward
#     # Note: The original distribution is not altered during the rollouts
#     new_position, updated_distribution, _ = simulate_step(size_x, size_y, cur_pos, best_action, obs, fov_dict, obstacles, temp_distribution, boundary_penalty)
#     return [best_action], updated_distribution, max_path_reward










# def receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, fov_dict, obstacles, last_action=None, horizon=5, rollout=5, epsilon=0.1, boundary_penalty=-0.1):
#     best_action = None
#     max_reward = -np.inf

#     for _ in range(rollout):
#         cumulative_reward = 0  # Reset cumulative reward for this rollout
#         temp_pos = cur_pos[:]  # Make a copy of the current position for this rollout

#         # The initial distribution is copied for each rollout to ensure it's not updated within the loop
#         temp_distribution_copy = np.copy(temp_distribution)

#         for step in range(horizon):
#             if np.random.rand() < epsilon:  # Exploration: choose a random action
#                 action = np.random.randint(0, 4)
#             else:  # Exploitation: choose the best action based on the current temporary distribution
#                 action_rewards = []
#                 for action_option in range(4):  # Evaluate all possible actions
#                     # Use a copy of the temp distribution to simulate the step without altering the original
#                     temp_distribution_for_step = np.copy(temp_distribution_copy)
#                     temp_pos, simulated_distribution, step_penalty = simulate_step(size_x, size_y, temp_pos, action_option, obs, fov_dict, obstacles, temp_distribution_for_step, boundary_penalty)
#                     # Calculate the reward for this action
#                     reward = simulated_distribution[temp_pos[0][0]][temp_pos[0][1]] + step_penalty
#                     action_rewards.append(reward)

#                 # Choose the action with the highest reward
#                 action = np.argmax(action_rewards)

#             # Simulate the step with the chosen action using the temporary distribution
#             temp_pos, temp_distribution_copy, step_penalty = simulate_step(size_x, size_y, temp_pos, action, obs, fov_dict, obstacles, temp_distribution_copy, boundary_penalty)
#             # Calculate the reward for the chosen action and update the cumulative reward
#             step_reward = temp_distribution_copy[temp_pos[0][0]][temp_pos[0][1]] + step_penalty
#             cumulative_reward += step_reward

#         # Update the best action and max reward if this rollout's cumulative reward is the highest
#         if cumulative_reward > max_reward:
#             max_reward = cumulative_reward
#             best_action = action  # Note: This should ideally be the first action of the best rollout

#     # Once the best action is determined, simulate the step with the original distribution to update it
#     new_position, updated_distribution, _ = simulate_step(size_x, size_y, cur_pos, best_action, obs, fov_dict, obstacles, temp_distribution, boundary_penalty)

#     # Return the best action, the updated distribution after taking that action, and the max reward from the best rollout
#     return [best_action], updated_distribution, max_reward



# def receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, fov_dict, obstacles, last_action=None, horizon=5, rollout=5, epsilon=0.1, boundary_penalty=-0.1):
#     best_path = []
#     max_reward = -np.inf

#     for _ in range(rollout):
#         temp_path = []
#         action_rewards_2 = []
#         total_reward = 0
#         temp_pos = cur_pos
#         temp_distribution_path = temp_distribution.copy()

#         for step in range(horizon):
#             if np.random.rand() < epsilon:  # Exploration
#                 action = np.random.randint(4)
#             else:  # Exploitation
#                 action_rewards = []
#                 for action_option in range(4):  # Evaluate all possible actions
#                     simulated_position, simulated_distribution, step_penalty = simulate_step(size_x, size_y, temp_pos, action_option, obs, fov_dict, obstacles, temp_distribution_path, boundary_penalty)
#                     reward = simulated_distribution[simulated_position[0][0]][simulated_position[0][1]] + step_penalty
#                     action_rewards.append(reward)
#                 print(action_rewards)

#                 action = np.argmax(action_rewards)  # Choose the action with the highest reward


#             # Simulate the step with the chosen action
#             temp_pos, temp_distribution_path, step_penalty = simulate_step(size_x, size_y, temp_pos, action, obs, fov_dict, obstacles, temp_distribution_path, boundary_penalty)
#             temp_path.append(action)
#             total_reward += temp_distribution_path[temp_pos[0][0]][temp_pos[0][1]] + step_penalty

#         print('Total reward: ', total_reward)
#         print('Path: ', temp_path)
#         if total_reward > max_reward:
#             max_reward = total_reward
#             best_path = temp_path

#         best_action = best_path[0]

#         print('Action: ', best_action, 'Reward: ', max_reward)
#     return [best_action], temp_distribution, max_reward
