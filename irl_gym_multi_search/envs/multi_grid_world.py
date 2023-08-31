"""
This module contains the GridworldEnv for discrete path planning
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = ""

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from copy import deepcopy
import logging

import numpy as np
from gymnasium import Env, spaces
import pygame
import matplotlib.cm as cm
from PIL import Image  # Import Image module from PIL


class multi_GridWorldEnv(Env):
    metadata = {"render_modes": ["plot", "print", "none"], "render_fps": 20}

    def __init__(self, *, seed : int = None, params : dict = None, start_pos, start_dir, fov_dict, obstacles, grid_size, num_agents):
        super(multi_GridWorldEnv, self).__init__()

        self._params = params if params else {}
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.start_positions = start_pos
        self.start_directions = start_dir
        self.start_dir = start_dir
        self.obstacles = obstacles
        self.fov_shape = fov_dict["fov_size"]

        self._id_action = {
            0: np.array([0, -1]), # up
            1: np.array([-1, 0]), # left
            2: np.array([0, 1]),  # down
            3: np.array([1, 0]),  # right
            4: np.array([0, 0]),  # dont move
        }
        self.action_space = spaces.discrete.Discrete(4)

        single_agent_observation_space = spaces.Dict(
            {
                "pose": spaces.box.Box(low=np.zeros(2), high=np.array(self.grid_size)-1, dtype=int),
                "obs": spaces.Box(low=0, high=1, shape=(1,), dtype=int),
                "orientation": spaces.Discrete(4),
                "rotated_fov": spaces.Box(low=0, high=1, shape=self.fov_shape, dtype=int)
            }
        )
        self.observation_space = spaces.Dict({agent_id: single_agent_observation_space for agent_id in range(self.num_agents)})

        self.agent_positions = {_: None for _ in range(self.num_agents)}
    

    def reset(self, *, seed: int = None, options: dict = {}):
        super().reset(seed=seed)
        self.steps = 0

        if self._params["render"] == "plot":
            color_bar_width = 60
            window_width = self._params["dimensions"][0] * self._params["cell_size"] + color_bar_width
            window_height = self._params["dimensions"][1] * self._params["cell_size"]

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_width, window_height))
            self.clock = pygame.time.Clock()

        if options != {}:
            for el in options:
                self._params[el] = deepcopy(options[el])
        
            if "dimensions" not in self._params:
                self._params["dimensions"] = [40, 40]
            if "goal" not in self._params:
                self._params["goal"] = [np.round(self._params["dimensions"][0]/4), np.round(self._params["dimensions"][1]/4)]
            if type(self._params["goal"]) != np.ndarray:
                self._params["goal"] = np.array(self._params["goal"], dtype = int)
            if "r_radius" not in self._params:
                self._params["r_radius"] = 5
            if "r_range" not in self._params:
                self.reward_range = (-0.01, 1)
            else:
                self.reward_range = self._params["r_range"]
            if "p" not in self._params:
                self._params["p"] = 0.1
            if "p_false_pos" not in self._params:
                self._params["p_false_pos"] = 0.1
            if "p_false_neg" not in self._params:
                self._params["p_false_neg"] = 0.1
            if "render" not in self._params:
                self._params["render"] = "none"
            if "print" not in self._params:
                self._params["print"] = False
            if self._params["render"] == "plot":
                self.window = None
                self.clock = None
                if "cell_size" not in self._params:
                    self._params["cell_size"] = 5
            if "save_frames" not in self._params:
                self._params["save_frames"] = False
            if "prefix" not in self._params:
                self._params["prefix"] = os.getcwd() + "/plot/"
            if self._params["save_frames"]:
                self._img_count = 0      

        if self._params["render"] == "plot":
            self._goal_polygons = [
                [
                    (goal + np.array([1, 0.5])) * self._params["cell_size"],
                    (goal + np.array([0.5, 1])) * self._params["cell_size"],
                    (goal + np.array([0, 0.5])) * self._params["cell_size"],
                    (goal + np.array([0.5, 0])) * self._params["cell_size"],
                ]
                for goal in self._params["goal"]
            ]
        self.agent_positions = {agent_id: {"pose": list(pos)} for agent_id, pos in enumerate(self.start_positions)}  # reset the agent positions to the start positions
        self.agent_orientations = [{'dir': direction} for direction in self.start_dir]

        self.found_goals = [False] * len(self._params["goal"]) # a list of the found goals

        return self.get_obs(fov=self._params["fov_dict"]["fov"])
    

    def get_obs(self, fov):
        """
        Generate observations for each position within the FOV.

        :param fov: (list) FOV matrix
        :return: (dict) Dictionary of observations for each agent
        """

        observations = {}

        # Mapping of orientation to the number of 90-degree rotations required
        orientation_mapping = {'u': 0, 'r': 1, 'd': 2, 'l': 3}

        for agent_id in range(self.num_agents):
            agent_position = self.agent_positions[agent_id]["pose"]
            agent_orientation = self.agent_orientations[agent_id]["dir"]
            
            # Determine the number of 90-degree rotations required based on agent orientation
            rotation_degree = orientation_mapping[agent_orientation]

            # Rotate the FOV by the required amount
            rotated_fov = np.rot90(fov, k=rotation_degree)

            agent_fov_x, agent_fov_y = self._params["fov_dict"]["fov_agent_position"]
            fov_width, fov_height = self._params["fov_dict"]["fov_size"]
            agent_x, agent_y = agent_position
            fov_x_start = agent_x - agent_fov_x
            fov_x_end = fov_x_start + fov_width
            fov_y_start = agent_y - agent_fov_y
            fov_y_end = fov_y_start + fov_height

            obs_fov = np.zeros_like(rotated_fov, dtype=bool) # Change to rotated_fov

            # Fill the FOV observation with the actual observations
            for y in range(fov_y_start, fov_y_end):
                for x in range(fov_x_start, fov_x_end):
                    fov_x_relative = x - fov_x_start
                    fov_y_relative = y - fov_y_start

                    # Check if the current cell is part of the goal and not found yet
                    cell_pos = [x, y]
                    is_goal = cell_pos in self._params["goal"]
                    goal_idx = self._params["goal"].index(cell_pos) if is_goal else None
                    is_goal_not_found = is_goal and not self.found_goals[goal_idx]

                    # If the current FOV value is 1 and matches with an unfound goal position
                    if rotated_fov[fov_y_relative][fov_x_relative] == 1 and is_goal_not_found: # Change to rotated_fov
                        # Mark goal as found
                        self.found_goals[goal_idx] = True
                        # True positive case (object is there and sensor says it's there)
                        cur_obs = np.random.random() > self._params["p_false_neg"]
                    else:
                        # True negative case (object is not there and sensor says it's not)
                        # OR False positive case (object is not there but sensor says it is)
                        cur_obs = np.random.random() < self._params["p_false_pos"]

                    obs_fov[fov_y_relative, fov_x_relative] = cur_obs
            observations[agent_id] = {
                "pose": agent_position,
                "obs": obs_fov,
                "rotated_fov": rotated_fov,
                "orientation": orientation_mapping[agent_orientation]}

        return observations
    

    def step(self, actions):
        """
        Increments environment by one timestep 
        
        :param actions: (list) list of actions for each agent, *default*: None
        :return: (tuple) State, reward, is_done, is_truncated, info 
        """
        assert len(actions) == self.num_agents, "Number of actions should be equal to the number of agents"

        # Define action to direction mapping. Makes each agent move forward (face in the direction of their action)
        action_to_dir = {0: 'u', 1: 'r', 2: 'd', 3: 'l', 4: 'u'}  # if 'dont move' is selected, then the agent looks up or north

        self.steps += 1
        reward = 0  # replace with your own reward calculation
        done = False  # replace with your own termination condition
        info = {}  # replace with your own info dict if needed
        truncated = False  # replace with your own condition for episode truncation

        # Generate a random order of agents
        agent_order = np.random.permutation(self.num_agents)

        # Iterate over agents in a random order
        for agent_id in agent_order:
            action = actions[agent_id]
            old_agent_state = self.agent_positions[agent_id]

            # Determine new position based on action
            p1 = old_agent_state["pose"].copy()
            p1 += self._id_action[action]

            # Update the agent's orientation based on the action
            self.agent_orientations[agent_id]["dir"] = action_to_dir[action]

            if (0 <= p1[0] <= (self._params["dimensions"][0] - 1) and 
                0 <= p1[1] <= (self._params["dimensions"][1] - 1)): # Make sure the agent is within the environment
                if self.obstacles[p1[0], p1[1]] < 0.5: # Avoid obstacle collisions
                    collision = False
                    for other_id, other_state in self.agent_positions.items():
                        if other_id != agent_id and np.all(p1 == other_state["pose"]):  # Prevent agent from occupying the same space as another agent
                            collision = True
                            break
                    if not collision:
                        self.agent_positions[agent_id]["pose"] = p1  # Update the pose only if there's no collision

        # Get observation after the action is performed
        observation = self.get_obs(fov=self._params["fov_dict"]["fov"])

        # Check if any agent is at the goal position and has a positive observation
        for agent_id in range(self.num_agents):
            agent_pos = self.agent_positions[agent_id]["pose"]
            positive_observation = observation[agent_id]
            
            for goal_idx, goal_pos in enumerate(self._params["goal"]):
                at_goal = np.all(agent_pos == goal_pos)
                if at_goal and positive_observation and not self.found_goals[goal_idx]:
                    self.found_goals[goal_idx] = True
                    break

        # If all goals are found, the simulation is done
        done = all(self.found_goals)

        reward = 0  # Replace with your own reward calculation
        info = {}  # Replace with your own info dict if needed
        truncated = False  # Replace with your own condition for episode truncation

        return observation, reward, done, truncated, info


    def custom_render(self, search_distribution, obstacles, region_outlines, return_frame=False):
        if self._params["render"] == "plot":
            color_bar_width = 60
            window_width = self._params["dimensions"][0] * self._params["cell_size"] + color_bar_width
            window_height = self._params["dimensions"][1] * self._params["cell_size"]

            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((window_width, window_height))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            img = pygame.Surface((self._params["dimensions"][0], self._params["dimensions"][1]))

            # Plot distribution
            normalized_distribution = (search_distribution - np.min(search_distribution)) / (np.max(search_distribution) - np.min(search_distribution))
            cmap = cm.get_cmap('viridis')
            for i in range(self._params["dimensions"][0]):
                for j in range(self._params["dimensions"][1]):
                    if obstacles[i, j] < 0.5:  # if the current square is in free space
                        color = cmap(normalized_distribution[i, j])
                        r, g, b, a = [int(c * 255) for c in color]
                    else:  # if an obstacle is in the current square
                        r, g, b, a = [30, 30, 30, 0]
                    img.set_at((i, j), (r, g, b, a))

            img = pygame.transform.scale(img, (self._params["dimensions"][0] * self._params["cell_size"], self._params["dimensions"][1] * self._params["cell_size"]))

            # Plot gridlines
            for y in range(self._params["dimensions"][1]):
                pygame.draw.line(img, 0, (0, self._params["cell_size"] * y), (self._params["cell_size"] * self._params["dimensions"][0], self._params["cell_size"] * y), width=1)
            for x in range(self._params["dimensions"][0]):
                pygame.draw.line(img, 0, (self._params["cell_size"] * x, 0), (self._params["cell_size"] * x, self._params["cell_size"] * self._params["dimensions"][1]), width=1)

            # Plot region outlines
            for outline in region_outlines:
                for contour in outline:
                    scaled_contour = ((contour + 0.5) * self._params["cell_size"]).astype(int)
                    pygame.draw.lines(img, (255, 0, 0), False, scaled_contour[:, [1, 0]], 1)

            # Render Goals
            for goal_idx, goal_polygon in enumerate(self._goal_polygons):
                color = (0, 255, 0) if self.found_goals[goal_idx] else (255, 0, 0)
                pygame.draw.polygon(img, color, goal_polygon)

            # Render Agents
            for agent_id in range(self.num_agents):
                agent_pos = self.agent_positions[agent_id]
                goal_pos = self._params["goal"]

                if any(np.all(agent_pos == goal_pos) for goal_pos in self._params["goal"]):
                    pygame.draw.polygon(img, (255, 255, 0), self._goal_polygon(agent_id))
                else:
                    pygame.draw.circle(img, (255, 255, 255), (np.array(agent_pos["pose"]) + 0.5) * self._params["cell_size"], self._params["cell_size"] / 2)

            # Create a surface for the color bar and the final surface
            color_bar_surface = pygame.Surface((color_bar_width, self._params["dimensions"][1] * self._params["cell_size"]))
            final_surface = pygame.Surface((window_width, window_height))
            cmap = cm.get_cmap('viridis')

            # Fill the color bar surface with a gradient corresponding to the colormap
            for y in range(color_bar_surface.get_height()):
                color_value = 1 - y / (color_bar_surface.get_height() - 1)  # Reverse the color gradient
                color = cmap(color_value)
                r, g, b, a = [int(c * 255) for c in color]
                pygame.draw.line(color_bar_surface, (r, g, b), (0, y), (color_bar_surface.get_width(), y))

            
            # Draw a gray vertical line to separate the grid from the color bar
            pygame.draw.line(color_bar_surface, (55, 55, 55), (0, 0), (0, self._params["dimensions"][1] * self._params["cell_size"]), 28)

            # Min and Max Values of the color bar
            min_value = np.min(search_distribution)
            max_value = np.max(search_distribution)

            # Calculate the step for the labels
            num_labels = 5
            value_step = (max_value - min_value) / (num_labels - 1)

            font = pygame.font.Font(None, 18)

            padding = 8  # Add some padding to avoid cutting off labels

            for i in range(num_labels):
                value = max_value - i * value_step  # Reverse the order of the labels
                y = padding + i * (color_bar_surface.get_height() - 2 * padding - 1) / (num_labels - 1)
                label_surface = font.render(str(np.round(value, 4)), True, (0, 0, 0))
                x = 18
                color_bar_surface.blit(label_surface, (x, y - label_surface.get_height() / 2))  # Blit to color_bar_surface

            # Combine img and color_bar_surface
            final_surface.blit(img, (0, 0))
            final_surface.blit(color_bar_surface, (self._params["dimensions"][0] * self._params["cell_size"], 0))

            # Render the final surface on the window
            self.window.blit(final_surface, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self._params["render_fps"])

            if return_frame:
                # Capture the screen content as a NumPy array
                pixels = pygame.surfarray.array3d(pygame.display.get_surface())
                # Transpose to fit the imageio format
                pixels = pixels.transpose([1, 0, 2])
                return pixels
        elif self._params["render"] == "print":
            pass


