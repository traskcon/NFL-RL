import functools
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv

class MultiEnvironment(ParallelEnv):
    metadata = {"name":"multiagent_environment-v0"}

    def __init__(self):
        '''Initial WR-DB Coverage Battle environment
        Based on PettingZoo prison escape tutorial: https://pettingzoo.farama.org/tutorials/custom_environment/2-environment-logic/
        '''
        #TODO: Convert locations into np arrays, like in single agent environment
        self.target_y = None
        self.target_x = None
        self.db_y = None
        self.db_x = None
        self.wr_y = None
        self.wr_x = None
        self.timestep = None
        self.possible_agents = ["WR", "DB"]

    def reset(self, seed=None, options=None):
        '''Re-initialize the environment'''
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.wr_x = 0
        self.wr_y = 0

        self.db_x = 6
        self.db_y = 6

        self.target_x = np.random.randint(2,5)
        self.target_y = np.random.randint(2,5)

        observation = (
            self.wr_x + 7 * self.wr_y,
            self.db_x + 7 * self.db_y,
            self.target_x + 7 * self.target_y,
        )
        observations = {
            "WR": {"WR": observation, "action_mask": [0,1,1,0]},
            "DB": {"DB": observation, "action_mask": [1,0,0,1]},
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        '''Take in action for given agent, update relevant states'''
        # Execute actions
        wr_action = actions["WR"]
        db_action = actions["DB"]

        if wr_action == 0 and self.wr_x > 0:
            self.wr_x -= 1
        elif wr_action == 1 and self.wr_x < 6:
            self.wr_x += 1
        elif wr_action == 2 and self.wr_y > 0:
            self.wr_y -= 1
        elif wr_action == 3 and self.wr_y < 6:
            self.wr_y += 1

        if db_action == 0 and self.db_x > 0:
            self.db_x -= 1
        elif db_action == 1 and self.db_x < 6:
            self.db_x += 1
        elif db_action == 2 and self.db_y > 0:
            self.db_y -= 1
        elif db_action == 3 and self.db_y < 6:
            self.db_y += 1

        # Generate action masks
        wr_action_mask = np.ones(4, dtype=np.int8)
        if self.wr_x == 0:
            wr_action_mask[0] = 0  # Block left movement
        elif self.wr_x == 6:
            wr_action_mask[1] = 0  # Block right movement
        if self.wr_y == 0:
            wr_action_mask[2] = 0  # Block down movement
        elif self.wr_y == 6:
            wr_action_mask[3] = 0  # Block up movement

        db_action_mask = np.ones(4, dtype=np.int8)
        if self.db_x == 0:
            db_action_mask[0] = 0
        elif self.db_x == 6:
            db_action_mask[1] = 0
        if self.db_y == 0:
            db_action_mask[2] = 0
        elif self.db_y == 6:
            db_action_mask[3] = 0

        # Action mask to prevent db from going over target cell
        if self.db_x - 1 == self.target_x:
            db_action_mask[0] = 0
        elif self.db_x + 1 == self.target_x:
            db_action_mask[1] = 0
        if self.db_y - 1 == self.target_y:
            db_action_mask[2] = 0
        elif self.db_y + 1 == self.target_y:
            db_action_mask[3] = 0

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.wr_x == self.target_x and self.wr_y == self.target_y:
            rewards = {"WR": 1, "DB": -1}
            terminations = {a: True for a in self.agents}
        elif self.wr_x == self.db_x and self.wr_y == self.db_y:
            rewards = {"WR": -1, "DB": 1}
            terminations = {a: True for a in self.agents}
        
        # Check truncation conditions
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"WR": 0, "DB": 0}
            truncations = {"WR": True, "DB": True}
        self.timestep += 1

        # Get observations
        observation = (
            self.wr_x + 7 * self.wr_y,
            self.db_x + 7 * self.db_y,
            self.target_x + 7 * self.target_y,
        )
        observations = {
            "WR": {"observation": observation, "action_mask": wr_action_mask,},
            "DB": {"observation": observation, "action_mask": db_action_mask,},
        }

        # Get dummy infos
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        '''Render the environment'''
        grid = np.full((7,7), ' ')
        grid[self.wr_y, self.wr_x] = "WR"
        grid[self.db_y, self.db_x] = "DB"
        grid[self.target_y, self.target_x] = "T"
        print(f"{grid} \n")

    #Define observation space
    #This line saves clock cycles, delete if spaces change over time
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([7 * 7 - 1] * 3)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)