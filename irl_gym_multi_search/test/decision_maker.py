
# import sys
import numpy as np
import random

# from search_algorithms.receding_horizon_search import receding_horizon_search
from search_algorithms.greedy_search_cells import greedy_search_cells
from search_algorithms.event_horizon_search import event_horizon
from search_algorithms.simulated_search import SimBFS
from search_algorithms.ra_ucb_search import RA_UCB
from search_algorithms.beam_search import beam_search
from search_algorithms.depth_limited_search import DLS
from search_algorithms.mcts_search import MCTS
from search_algorithms.mcts_search_region import MCTS_Region
from search_algorithms.mcts_region_search import MCTSWithRegions
from search_algorithms.greedy_region_search import RegionGreedySearch
from search_algorithms.regionMCTS_search import RegionMCTSSearch


def decision_maker_init(distribution, regions, fov_dict):
    # greedy_region_search = RegionGreedySearch()
    mcts_region_search = RegionMCTSSearch()
    return mcts_region_search


# Takes in the probability density at each location using the search distribution

def search_decision_maker(size_x, size_y, cur_pos, observation, obstacles, last_action, path, distribution, fov_dict, current_fov, cur_orientation, waypoint_reached, waypoint, regions, mcts_region_search):

    # if waypoint is True: (this means that the agent is currently traveling to a waypoint)
        # skip over search algs and only do relative pos update. The agent's current position and the waypoint position will go into this.
        # The waypoint position will come from region mcts if the action > 4 (which would be a move to region action). It should find the closest value to the agent within the selected region. This is the waypoint.
        # mcts should also return waypoint_active (True/False), and waypoint (actual position of the waypoint). 

    ########################################
    # Search methods

    # relative_pos = greedy_search_cells(cur_pos, distribution) # do greedy search for individual cells

    # Event horizon search
    # relative_pos = event_horizon(cur_pos, distribution)

    # Receding horizon search
    # best_action, visited, reward = receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, observation, last_action, horizon=40, rollout=5, epsilon=0.1)
    # relative_pos = np.array(cur_pos) - np.array(cur_pos)
    # print(reward)

    # BFS event horizon search
    # next_pos = SimBFS(size_x, size_y, cur_pos, distribution, steps=6)
    # relative_pos = np.array(next_pos) - np.array(cur_pos)

    # Rollout Allocation using Upper Confidence Bounds (RA-UCB)
    # next_pos = RA_UCB(size_x, size_y, cur_pos, distribution, steps=15, num_samples=100)
    # relative_pos = np.array(next_pos) - np.array(cur_pos)

    # Beam search
    # next_pos = beam_search(size_x, size_y, cur_pos, distribution, steps=25, beam_width=20)
    # relative_pos = np.array(next_pos) - np.array(cur_pos)

    # Depth limited search
    # next_pos = DLS(size_x, size_y, cur_pos, distribution, limit=4)
    # relative_pos = np.array(next_pos) - np.array(cur_pos)

    # mcts = MCTS(size_x, size_y, cur_pos, distribution, obstacles, num_iterations=90, c_param=0.2, max_rollout_steps=48)
    # best_action = mcts.simulate() # Run the MCTS to get the optimal action
    # relative_pos = np.array(cur_pos) - np.array(cur_pos)

    if path is not None and (len(path) > 1 or (len(path) == 1 and path[0] is None)): # if there is any leftover path from the previous MCTS run
        path.pop(0) # remove the first action within the path
        best_path = path # update the path to be the current path
    else:
        region_action_modifier = 0.0036 / (size_x * size_y)
        mcts_region = MCTS_Region(size_x, size_y, cur_pos, distribution, regions, fov_dict, current_fov, cur_orientation, last_action, obstacles,
                                  num_iterations=165, c_param=0.3, max_rollout_steps=42, region_action_modifier=region_action_modifier)
        best_path = mcts_region.simulate()
    
    print(best_path)

    # relative_pos = np.array(cur_pos) - np.array(cur_pos)

    # mcts_region = MCTSWithRegions(regions, size_x, size_y, cur_pos, distribution, obstacles, num_iterations=300, c_param=0.6, max_rollout_steps=10)
    # best_action = mcts_region.simulate()
    # print(best_action)
    # relative_pos = np.array(cur_pos) - np.array(cur_pos)


    # best_action = mcts_region_search.select_next_action(cur_pos, distribution, fov_dict, current_fov, regions)
    # relative_pos = np.array(cur_pos) - np.array(cur_pos)


    # best_action = greedy_region_search.select_next_action(cur_pos, distribution, fov_dict, current_fov, regions)

    # relative_pos = np.array(cur_pos) - np.array(cur_pos)

    # Perform random search (go to a random cell)
    # if waypoint_reached is True:
    #     # Compute random indices
    #     i = np.random.randint(0, distribution.shape[0])
    #     j = np.random.randint(0, distribution.shape[1])

    #     waypoint[0] = i
    #     waypoint[1] = j
    #     # Compute the relative position from the current position to the maximum probability density
    
    # relative_pos = np.array([i, j]) - cur_pos

    # Perform greedy cell search
    # relative_pos = greedy_search_cells(cur_pos, distribution)

    ########################################

    # Choose an action - go to the desired cell
    # Choose the action that minimizes the relative position in both x and y axes
    # From gym env:
    # 0: np.array([0, -1]), # up
    # 1: np.array([-1, 0]), # left
    # 2: np.array([0, 1]),  # down
    # 3: np.array([1, 0]),  # right

    # if np.abs(relative_pos[0]) >= np.abs(relative_pos[1]):
    #     if relative_pos[0] < 0:
    #         action = 1  # Move left
    #     else:
    #         action = 3  # Move right
    # else:
    #     if relative_pos[1] < 0:
    #         action = 0  # Move up
    #     else:
    #         action = 2  # Move down
    # if relative_pos[0] == 0 and relative_pos[1] == 0:
    #     action = 4

    # if np.abs(relative_pos[0]) < 1 and np.abs(relative_pos[1]) < 1:
    #     waypoint_reached = True
    # else:
    #     waypoint_reached = False


    # action = best_action
    action = best_path
    
    return action, distribution, waypoint_reached, waypoint
