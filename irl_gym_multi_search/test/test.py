import gymnasium as gym
import numpy as np
import numpy.random as r
import imageio
import pickle
from copy import deepcopy
from view_results import view_results
from decision_maker import search_decision_maker
from decision_maker import decision_maker_init
from probability_distribution import gaussian_filter
from obstacles import obstacles0, obstacles1, obstacles2
from segment_regions import segment_regions
from irl_gym_multi_search.test.update_distribution import update_distribution
import datetime
import time
import copy

##############################################
# Adjustable Parameters/Settings:
# ------------------------------
# Field of view (size and shape)
# Render FPS
# Record and save window to gif
# Record and save data
# Number of trials
# RNG seed
# Size of environment in x
# Size of environment in y
# Start positions for each agent (number of agents is derived from number of start positions given) 
# Start directions 'u', 'd', 'l', 'r'
# Number of lost objects
# Probability peak number
# Probability peak height
# Probability peak width
# Probability peak length
# Probability peak rotation
# Minimum region peak value
# Minimum region cell value
# Obstacles
# Cell size (for rendering)
# False positive value for sensor
# False negative value for sensor
# Max episode steps
##############################################



# represents the field of view for an agent. Each value represents a single cell.
# 1 is observable, 0 is not. The middle cell will be the agent's position within this grid. The size in both directions must be an odd number to ensure a center point.
# this must also be square to be rotated. This assumes the start direction is up
fov = [[1, 1, 1],
       [1, 0, 1],
       [1, 1, 1]]

# fov = [[0, 1, 1, 1, 0],
#        [1, 1, 1, 1, 1],
#        [1, 1, 0, 1, 1],
#        [1, 1, 1, 1, 1],
#        [0, 1, 1, 1, 0]]

# fov = [[0, 0, 0],
#        [0, 1, 0],
#        [0, 0, 0]]

# fov = [[1]]

fov_size = [len(fov[0]), len(fov)] # gets the field of view size [x, y]
fov_agent_x = fov_size[0] // 2
fov_agent_y = fov_size[1] // 2 
fov_agent_position = [fov_agent_x, fov_agent_y]

fov_dict = {
    "fov": fov,
    "fov_size": fov_size,
    "fov_agent_position": fov_agent_position
}

cell_size = 16
render_fps = 60  # max fps
render_fps_gif = 20  # gif fps

render = True
save_to_gif = False
save_data = True

num_trials = 1  # number of trials

seed = 3
rng = r.default_rng(seed=seed)

max_steps = 2500

# size of environment
size_x = 80
size_y = 70

# Initialize start positions and directions
start = [[30, 48]]
start_dir = ['u']

start_dir_copy = copy.deepcopy(start_dir)
num_agents = len(start)

# probability values for the agent's sensor
p_false_pos = 0.0 # 0 is perfect, towards 1 is noisy
p_false_neg = 0.0 # 0 is perfect, towards 1 is noisy

# number of lost objects
num_objects = 4

# Probability Distributions
# define the ranges for random values
num_peaks_range = [num_objects, num_objects + 3]  # range for number of peaks (could be overlapping peaks)
peak_height_range = [0.18, 0.88]  # range for peak heights. values in probability within a square
peak_width_range_x = [0.04, 0.12]  # range for peak width in x. values in % of total environment size
peak_width_range_y = [0.04, 0.12]  # range for peak width in y. values in % of total environment size
peak_rot_range = [0, np.deg2rad(180)]  # range for peak orientation

minRegionPeak = 1.2 / (size_x * size_y) # specifies the minimum value for a cell to be considered as a peak of a distribution
minRegionValue = minRegionPeak # specifies the minimum value for a cell to be within a region (must be <= minRegionPeak)

# create the search distribution
search_distribution, peaks, centers = gaussian_filter(seed, size_x, size_y, num_objects, num_peaks_range, peak_height_range, peak_width_range_x, peak_width_range_y, peak_rot_range)
# search_distribution = np.full((size_x, size_y), 0.5) # creates a uniform search distribution

# create obstacles within the environment (call 'obstacles0' for no obstacles)
obstacles, obstacles_name = obstacles0(size_x, size_y, start)
obstacles = obstacles.astype(int)
search_distribution = search_distribution * ~obstacles.astype(bool) # set the search distribution cells to zero where there are obstacles (cannot have an object where there is an obstacle)

# Generate regions of interest (ROI's) from the search distribution
regions, region_outlines = segment_regions(search_distribution, minRegionPeak, minRegionValue)

# Print the regions information
print('Region, x, y, num pts, weight, total')
for region in regions:
    print("[{:4.0f} {:3.0f} {:3.0f} {:5.0f} {:7.4f} {:7.4f}]".format(
        region['index'],       # region number
        region['peak_x'],      # peak value x coordinate
        region['peak_y'],      # peak value y coordinate
        region['num_points'],  # total number of cells within the region
        region['weight'],      # average distribution value within the region
        region['total']        # sum of distribution values within the region
    ))# region['points'] also exists, and represents the xy coords for every cell within the region

# Get new goal positions
selected_regions = rng.choice(regions, min(num_objects, len(regions)), replace=False)
goal = []
for region in selected_regions:
    points = region['points']
    valid_goal = False # Keep trying until a valid goal is found
    while not valid_goal:
        x, y = rng.choice(points) # Randomly select a point within the region
        if not any([x == s[0] and y == s[1] for s in start]): # Check if the goal is a start position
            goal.append([x, y])
            valid_goal = True

num_objects = len(goal) # Update the number of objects (should remain the same)

if render:
    render_param = "plot"
else:
    render_param = "none"

param = { # create dictionary of parameters for the gym environment
    "render": render_param,
    "render_fps": render_fps,
    "dimensions": [size_x, size_y],
    "cell_size": cell_size,
    "goal": [list(g) for g in goal],
    "state": {agent_id: {"pose": pos} for agent_id, pos in enumerate([list(s) for s in start])},
    "fov_dict": {
        "fov": fov,
        "fov_size": fov_size,
        "fov_agent_position": fov_agent_position
    },
    "p_false_pos": p_false_pos,
    "p_false_neg": p_false_neg
}

# create the gym environment
env = gym.make("irl_gym_multi_search/multi_GridWorldEnv-v0", max_episode_steps=max_steps, params=param, start_pos=start, start_dir=start_dir, fov_dict=fov_dict, obstacles=obstacles, grid_size=[size_x, size_y], num_agents=num_agents)

original_search_distribution = np.copy(search_distribution)

# save setup
setup = {"seed": seed, "size_x": size_x, "size_y": size_y, "num_agents": num_agents, "start_pos": start, "num_goals": num_objects,
         "goal": goal, "num_peaks_range": num_peaks_range,
         "peak_height_range": peak_height_range, "peak_width_range_x": peak_width_range_x,
         "peak_width_range_y": peak_width_range_y, "peak_rot_range": np.round(peak_rot_range,4),
         "obstacles": obstacles_name, "p_false_pos": param["p_false_pos"], "p_false_neg": param["p_false_neg"],
         "max_episode_steps": max_steps, "num_trials": num_trials, "fov": fov_dict["fov"], "fov_size": fov_dict["fov_size"]}

results = []  # store results from each trial
frames = []  # store rendered frames to save as a gif

overall_start_time = time.time() # start the timer
trial_times = [] # create a list of individual trial times

for i in range(num_trials): # loop through the number of trials
    trial_start_time = time.time() # start the timer for the active trial
    print(f'Starting trial {i+1}')

    mcts_region_search = decision_maker_init(search_distribution, regions, fov_dict) # initialize the decision maker (if needed)

    search_distribution = np.copy(original_search_distribution) # reset search distribution
    done = False # reset done criteria

    waypoint = [list(pos) for pos in start]  # each agent has its own waypoint
    waypoint_reached = [True for _ in range(num_agents)]  # each agent has its own waypoint_reached status
    visited = np.zeros((size_x, size_y), dtype=bool)
    action=[None for _ in range(num_agents)]  # each agent has its own action
    paths = [None] * num_agents
    current_position = [list(pos) for pos in start]  # each agent has its own current position
    cur_orientation = start_dir_copy
    print(current_position)
    search_observation = {agent_id: [] for agent_id in range(num_agents)}
    current_fov = {agent_id: [] for agent_id in range(num_agents)}
    step = 0
    env.reset() # reset the gym environment. must be done before every trial

    if render: # initial frame render
        frame = env.custom_render(search_distribution, obstacles, region_outlines, return_frame=True)  # render the environment with search probabilities
        frames.append(frame)
        time.sleep(0.5) # let the pygame window load and render before starting

    for agent_id in range(num_agents): # step through each agent
        # Get initial observation
        observation = env.get_obs(fov=fov_dict["fov"]) # get the current fov
        search_observation[agent_id] = observation[agent_id]['obs'] # Save the observation for the agent (whether a goal has been seen or not within the current fov)
        current_position[agent_id][0] = observation[agent_id]['pose'][0]  # Update x value
        current_position[agent_id][1] = observation[agent_id]['pose'][1]  # Update y value
        cur_orientation[agent_id] = observation[agent_id]['orientation']  # update orientation
        current_fov[agent_id] = observation[agent_id]['rotated_fov'] # update the fov (changes with orientation)

    # update the distribution with the initial observation(s)
    search_distribution = update_distribution(size_x, size_y, num_agents, current_position, current_fov, search_observation, fov_dict, obstacles, search_distribution)

    if render: # render the updated distribution
        frame = env.custom_render(search_distribution, obstacles, region_outlines, return_frame=True)  # render the environment with search probabilities
        frames.append(frame)

    while not done: # trial loop. this loop runs until the end of the single trial

        actions = rng.choice(4, size=num_agents)

        for agent_idx in range(num_agents): # loop through each agent
            paths[agent_idx], search_distribution, waypoint_reached[agent_idx], waypoint[agent_idx] = search_decision_maker( # call the search decision maker for the current agent
                size_x, size_y, current_position[agent_idx], search_observation[agent_idx], obstacles, action[agent_idx], paths[agent_idx], search_distribution, fov_dict, current_fov[agent_idx],
                cur_orientation[agent_idx], waypoint_reached[agent_idx], waypoint[agent_idx], regions, mcts_region_search)
            
            actions[agent_idx] = paths[agent_idx][0] # take the first action from the path given from the decision maker. Gym only accepts a single action at a time

        observation, r, done, is_trunc, info = env.step(actions) # simulate a single step with the enviroment. Each agent's action is also given here. The observation, stopping criteria, and extra info is returned
        print(f'{i+1} {step}')
        for agent_id in range(num_agents): # step through each agent
            search_observation[agent_id] = observation[agent_id]['obs'] # Save the observation for the agent (whether a goal has been seen or not within the current fov)
            current_position[agent_id][0] = observation[agent_id]['pose'][0]  # Update x value
            current_position[agent_id][1] = observation[agent_id]['pose'][1]  # Update y value
            cur_orientation[agent_id] = observation[agent_id]['orientation']  # update orientation
            current_fov[agent_id] = observation[agent_id]['rotated_fov'] # update the fov (changes with orientation)

        search_distribution = update_distribution(size_x, size_y, num_agents, current_position, current_fov, search_observation, fov_dict, obstacles, search_distribution)  # update the search distribution
        if render:
            frame = env.custom_render(search_distribution, obstacles, region_outlines, return_frame=True)  # render the environment with search probabilities
            frames.append(frame)
        step += 1 # increment step (gym also does this internally)
        done = done or is_trunc
        # end of trial loop for individual trials
    
    # once the current trial has finished, save its results and let the user know the outcome
    results.append((step, not is_trunc))  # save the number of steps and is_trunc status for each trial
    trial_end_time = time.time()
    trial_elapsed_time = trial_end_time - trial_start_time
    trial_times.append(trial_elapsed_time)
    print('Trial time (sec): ', trial_elapsed_time)
    if is_trunc == False:
        print(f'Trial {i+1}: Found goal after {step} steps')
    else:
        print(f'Trial {i+1}: Ran out of steps after {step} steps')

    # end of total trials loop

# once all of the trials have been completed, save the results and frames and let the user know
print(f'\nRan {num_trials} trials \n')

overall_end_time = time.time()
overall_elapsed_time = overall_end_time - overall_start_time

average_trial_time = overall_elapsed_time / num_trials

print('Total time (sec): ', overall_elapsed_time)
print('Average time (sec): ', average_trial_time)

# get current time to use in file names
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if render: # only save to gif if rendered
    if save_to_gif:
        gif_file_name = f'irl_gym_multi_search/test/gifs/recording_{current_time}.gif'
        print('Saving pygame frames to gif')
        imageio.mimsave(gif_file_name, frames, duration=1000/render_fps_gif, loop=0)
        print('Saved to gif')

if save_data:
    results_file_name = f'irl_gym_multi_search/test/experiment data/experiment_data_{current_time}.pickle'
    # save setup and results
    with open(results_file_name, 'wb') as f:
        pickle.dump((setup, results), f)

    # print results
    for i, result in enumerate(results):
        steps, is_trunc = result
        if is_trunc == True:
            print(f'Trial {i+1}: Goal Reached in {steps} steps')
        else:
            print(f'Trial {i+1}: Ran out of steps after {steps} steps')

    print(f'Setup and results saved within: {results_file_name}')

    view_results(results_file_name)
