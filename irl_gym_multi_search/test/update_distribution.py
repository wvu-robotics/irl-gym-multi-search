
import numpy as np

def update_distribution(size_x, size_y, num_agents, cur_pos, cur_fov, observation, fov_dict, obstacles, prior_distribution):
    # Start with a copy of the prior distribution
    new_distribution = prior_distribution.copy()

    for agent_id in range(num_agents):
        agent_position = cur_pos[agent_id]
        fov_offset_x, fov_offset_y = fov_dict["fov_agent_position"]  # Offset of the agent's position within the FOV
        fov_width, fov_height = fov_dict["fov_size"]

        agent_x, agent_y = agent_position
        fov_x_start = agent_x - fov_offset_x
        fov_y_start = agent_y - fov_offset_y

        # Track the total probability mass observed
        total_observed_prob = 0.0

        # Loop through the entire FOV, but only update cells that are inside the environment's boundaries
        for fov_x_relative in range(fov_width):
            for fov_y_relative in range(fov_height):
                x = fov_x_start + fov_x_relative
                y = fov_y_start + fov_y_relative

                if 0 <= x < size_x and 0 <= y < size_y and cur_fov[agent_id][fov_y_relative][fov_x_relative] == 1:
                    if observation[agent_id][fov_y_relative, fov_x_relative] == True:
                        total_observed_prob += new_distribution[x, y]
                        new_distribution[x, y] = 0.0
                    elif observation[agent_id][fov_y_relative, fov_x_relative] == False:
                        total_observed_prob += new_distribution[x, y]
                        new_distribution[x, y] = 0.0

        # Redistribute the observed probability mass proportionally to the existing probabilities
        redistribution_factor = total_observed_prob / (1 - total_observed_prob)
        new_distribution *= (1 + redistribution_factor)

        # Set the search distribution cells to zero where there are obstacles
        new_distribution *= ~obstacles.astype(bool)

        # Normalize the distribution
        new_distribution /= np.sum(new_distribution)

    return new_distribution

