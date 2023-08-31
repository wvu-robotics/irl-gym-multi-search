from gymnasium.envs.registration import register

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

"""
Installs irl_gym_multi_search envs

"""
register(
    id='irl_gym_multi_search/multi_GridWorldEnv-v0',
    entry_point='irl_gym_multi_search.envs:multi_GridWorldEnv',
    max_episode_steps=100,
    reward_threshold = None,
    disable_env_checker=False,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "dimensions": [40,40],
            "goal": [10,10],
            "state": 
            {
                "pose": [20,20],
            },
            "p_false_pos":0.1,
            "p_false_neg":0.1,
            "render": "none",
            "cell_size": 50,
            "prefix": current + "/plot/",
            "save_frames": False,
            "log_level": "WARNING"
        }
    }
)
