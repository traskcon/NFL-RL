import functools
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv

class MultiEnvironment(ParallelEnv):
    metadata = {"name":"multiagent_environment-v0"}

    def __init__(self, render_mode=None, width=124, length=57):
        '''Initial WR-DB Coverage Battle environment
        Based on PettingZoo prison escape tutorial: https://pettingzoo.farama.org/tutorials/custom_environment/2-environment-logic/
        '''
        scale_factor = 10
        self.width = width  # The width of the football field grid (53 "in-bounds" + 4 "out-of-bounds")
        self.length = length # The length of the football field grid (120 "in-bounds" + 4 "out-of-bounds")
        self.window_width = width*scale_factor  # The dimensions of the PyGame window
        self.window_length = length*scale_factor

        self.target_location = np.array([None, None])
        self.db_location = np.array([None, None])
        self.wr_location = np.array([None, None])
        self.timestep = None
        self.possible_agents = ["WR", "DB"]


        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        '''Re-initialize the environment'''
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.wr_location = np.array([0, 0])

        self.db_location = np.array([6, 6])

        self.target_location = np.random.randint(2,5,2)

        observations = {a:(
            np.dot([1,7], self.wr_location),
            np.dot([1,7], self.db_location),
            np.dot([1,7], self.target_location),
        ) for a in self.agents}

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        '''Take in action for given agent, update relevant states'''
        # Execute actions
        wr_action = actions["WR"]
        db_action = actions["DB"]

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        wr_direction = self._action_to_direction[wr_action]
        # We use `np.clip` to make sure we don't leave the grid
        self.wr_location = np.clip(
            self.wr_location + wr_direction, [0, 0], [self.width - 1, self.length - 1]
        )

        db_direction = self._action_to_direction[db_action]
        self.db_location = np.clip(
            self.db_location + db_direction, [0, 0], [self.width - 1, self.length - 1]
        )

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if np.array_equal(self.wr_location, self.target_location):
            rewards = {"WR": 1, "DB": -1}
            terminations = {a: True for a in self.agents}
        elif np.array_equal(self.wr_location, self.db_location):
            rewards = {"WR": -1, "DB": 1}
            terminations = {a: True for a in self.agents}
        
        # Check truncation conditions
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"WR": 0, "DB": 0}
            truncations = {"WR": True, "DB": True}
        self.timestep += 1

        # Get observations
        observations = {a: (
            np.dot([1,7], self.wr_location),
            np.dot([1,7], self.db_location),
            np.dot([1,7], self.target_location),
        ) for a in self.agents}

        # Get dummy infos
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        '''Render the environment'''
        grid = np.full((7,7), ' ')
        grid[self.wr_location[1], self.wr_location[0]] = "WR"
        grid[self.db_location[1], self.db_location[0]] = "DB"
        grid[self.target_location[1], self.target_location[0]] = "T"
        print(f"{grid} \n")

    #Define observation space
    #This line saves clock cycles, delete if spaces change over time
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([7 * 7 - 1] * 3)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)