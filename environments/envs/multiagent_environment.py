import functools
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium import spaces
import pygame
from environments.utils.core import Agent, Landmark, World
from pettingzoo import ParallelEnv

class MultiEnvironment(ParallelEnv):
    metadata = {"name":"multiagent_environment-v0", "render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, max_cycles, scenario, render_mode=None, width=124, length=57):
        '''Initial WR-DB Coverage Battle environment
        Based on PettingZoo prison escape tutorial: https://pettingzoo.farama.org/tutorials/custom_environment/2-environment-logic/
        '''
        # Initialize parameters for field, rendering
        scale_factor = 10
        self.width = width  # The width of the football field grid (53 "in-bounds" + 4 "out-of-bounds")
        self.length = length # The length of the football field grid (120 "in-bounds" + 4 "out-of-bounds")
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.window_width = width*scale_factor  # The dimensions of the PyGame window
        self.window_length = length*scale_factor

        self.target_location = np.array([None, None])
        self.timestep = None
        self.world = scenario.make_world()
        self.agent_names = [agent.name for agent in self.world.agents]
        self.colormap = {"WR_0":(0,0,255), "DB_0":(255,0,0)}

        self.observation_spaces = dict()
        self.action_spaces = dict()
        for name in self.agent_names:
            self.observation_spaces[name] = spaces.Box(low=np.array([0, 0]), high=np.array([width - 1, length - 1]), dtype=int)
            self.action_spaces[name] = spaces.Discrete(4)

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
        self.timestep = 0
        self.scenario.reset_world(self.world)

        self.rewards = {name: 0.0 for name in self.agent_names}
        self.target_location = self.world.agents[0].goal_a.location

        observations = {a.name:(
            [agent.location for agent in self.world.agents],
            self.target_location,
        ) for a in self.world.agents}

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {name: {} for name in self.agent_names}

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    def step(self, actions):
        '''Set action for all agents, update relevant states'''
        for agent in self.world.agents:
            agent_action = actions[agent.name] # Execute action
            agent_direction = self._action_to_direction[agent_action] #Map the action (0,1,2,3) to the direction of movement
            # Use np.clip to ensure we don't leave the grid
            agent.location = np.clip(
                agent.location + agent_direction, [0, 0], [self.width - 1, self.length - 1]
            )
    
        # Check termination conditions
        # TODO: Replace with relevant rewards, terminations
        terminations = {name: False for name in self.agent_names}
        rewards = {name: 0 for name in self.agent_names}

        # Check truncation conditions
        truncations = {name: False for name in self.agent_names}
        if self.timestep > self.max_cycles:
            self.rewards = {"WR": 0, "DB": 0}
            self.truncations = {"WR": True, "DB": True}
        self.timestep += 1

        # Get observations
        observations = {a.name: (
            [agent.location for agent in self.world.agents],
            self.target_location,
        ) for a in self.world.agents}

        # Get dummy infos
        infos = {name: {} for name in self.agent_names}

        if any(terminations.values()) or all(truncations.values()):
            # If termination/truncation condition met, remove all agents
            self.world.agents = []
        
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
        # Now we draw the Agents
        for agent in self.world.agents:
            pygame.draw.circle(
                canvas,
                self.colormap[agent.name],
                (agent.location + 0.5) * pix_square_size,
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
        return self.observation_spaces[agent.name]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent.name]
    

class Scenario():
    def make_world(self, N=2):
        world = World()
        num_agents = N
        world.num_agents = num_agents
        num_defense = N/2
        num_landmarks = 1
        positions = ["WR", "DB"]
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.defense = True if i < num_defense else False
            agent.position = positions[i]
            base_index = i if i < num_defense else int(i - num_defense)
            agent.name = f"{agent.position}_{base_index}"
            agent.location = np.array([None, None])
        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.location = np.array([None, None])
        return world

    def agent_reward(self, agent, world):
        # Reward WR by how close they are to landmark and how far DB is from them
        # Currently doing well = positive reward values
        defensive_players = self.defensive_players(world)
        def_rew = sum(np.sqrt(np.sum(np.square(a.location - agent.location)))
                      for a in defensive_players)
        off_rew = -np.sqrt(np.sum(np.square(agent.location - agent.goal_a.location)))
        return off_rew + def_rew
    
    def adversary_reward(self, agent, world):
        offensive_players = self.offensive_players(world)
        def_rew = -sum(np.sqrt(np.sum(np.square(agent.location - a.location)))
                      for a in offensive_players)
        return def_rew

    def defensive_players(self, world):
        return [agent for agent in world.agents if agent.defense]
    
    def offensive_players(self, world):
        return [agent for agent in world.agents if not agent.defense]
    
    def reset_world(self, world):
        # Initial positions for landmark, agents
        goal = np.random.choice(world.landmarks)
        # Can build formations as an argument here
        starting_locations = {"WR_0":np.array([20, 10]), "DB_0":np.array([60,16])}
        for agent in world.agents:
            agent.goal_a = goal
            agent.location = starting_locations[agent.name]
        for landmark in world.landmarks:
            # Can use landmarks to design plays for agents
            landmark.location = np.random.randint(15,45,2)