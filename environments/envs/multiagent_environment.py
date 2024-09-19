import functools
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium import spaces
import pygame

from pettingzoo import ParallelEnv

class MultiEnvironment(ParallelEnv):
    metadata = {"name":"multiagent_environment-v0", "render_modes": ["human", "rgb_array"], "render_fps": 4}

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

        self.observation_spaces = dict()
        self.action_spaces = dict()
        for agent in self.possible_agents:
            self.observation_spaces[agent] = spaces.Box(low=np.array([0, 0]), high=np.array([width - 1, length - 1]), dtype=int)
            self.action_spaces[agent] = spaces.Discrete(4)

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

        self.wr_location = np.array([20, 10])

        self.db_location = np.array([60, 16])

        self.target_location = np.random.randint(25,45,2)

        observations = {a:(
            self.wr_location,
            self.db_location,
            self.target_location,
        ) for a in self.agents}

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        if self.render_mode == "human":
            self._render_frame()

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
            self.wr_location,
            self.db_location,
            self.target_location,
        ) for a in self.agents}

        # Get dummy infos
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_length)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_length))
        canvas.fill((54, 155, 44)) #Color the field green
        pix_square_size = (
            self.window_width / self.width
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                pix_square_size * self.target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the WR (blue)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.wr_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Now we draw the DB (blue)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self.db_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        # Gridlines are very harsh on the eyes at this scale, but useful code example for future environment beautification
        for x in range(self.length + 1):
            pygame.draw.line(
                canvas,
                (0,0,0, 0.2),
                (0, pix_square_size * x),
                (self.window_width, pix_square_size * x),
                width=2,
            )
        for y in range(self.width + 1):
            pygame.draw.line(
                canvas,
                (0,0,0, 0.2),
                (pix_square_size * y, 0),
                (pix_square_size * y, self.window_length),
                width=2,
            )
        # Draw sidelines
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                (0, 0),
                (pix_square_size * self.window_width, pix_square_size*2),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                (0, self.window_length - 2*pix_square_size),
                (pix_square_size * self.window_width, pix_square_size*2),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                (0, 0),
                (2*pix_square_size, pix_square_size*self.window_length),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                (self.window_width - 2*pix_square_size, 0),
                (2*pix_square_size, pix_square_size*self.window_length),
            ),
        )
        # Draw Endzones
        pygame.draw.rect(
            canvas,
            (224, 220, 226),
            pygame.Rect(
                (2*pix_square_size, 2*pix_square_size),
                (10*pix_square_size, self.window_length - 4*pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 118, 182) , #Go Lions
            pygame.Rect(
                (self.window_width - 12*pix_square_size, 2*pix_square_size),
                (10*pix_square_size, self.window_length - 4*pix_square_size),
            ),
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    #Define observation space
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]