import pandas as pd
import numpy as np
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
        self.width = width  # The width of the football field grid (120 "in-bounds" + 4 "out-of-bounds")
        self.length = length # The length of the football field grid (53 "in-bounds" + 4 "out-of-bounds")
        self.max_cycles = max_cycles * 0.15 # Convert max cycles (# of iterations) to timescale
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
        self.world = self.scenario.make_world()
        self.scenario.reset_world(self.world)

        self.rewards = {name: 0.0 for name in self.agent_names}
        self.terminations = {name: False for name in self.agent_names}
        self.target_location = self.world.agents[0].goal_a.location

        observations = {a.name:(
            *[agent.location for agent in self.world.agents],
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
        for agent in self.world.agents:
            self.terminations[agent.name] = self.termination(agent)
            self.rewards[agent.name] = self.scenario.reward(agent, self.world)

        # Check truncation conditions
        truncations = {name: False for name in self.agent_names}
        if self.timestep > self.max_cycles:
            self.rewards = {name: 0 for name in self.agent_names}
            truncations = {name: True for name in self.agent_names}
        self.timestep += 0.15  # Timestep of 0.1s means players are assumed to be moving at 6.67 yards/s = 20 ft/s = 13.6 mph
        self.world.timestep += 0.15

        # Get observations
        observations = {a.name: (
            *[agent.location for agent in self.world.agents],
            self.target_location - a.location,
        ) for a in self.world.agents}

        # Get dummy infos
        infos = {name: {} for name in self.agent_names}

        if any(self.terminations.values()) or all(truncations.values()):
            # If termination/truncation condition met, remove all agents
            self.world.agents = []
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observations, self.rewards, self.terminations, truncations, infos
    
    def termination(self, agent):
        # Check if an offensive player has stepped out of bounds
        # TODO: Add code checking if ballcarrier scored a touchdown
        if not agent.defense:
            if np.sum(np.square(agent.location - agent.goal_a.location)) == 0:
                return True #End simulation if WR reaches target
            else:
                self.check_bounds(agent)
                return agent.oob
        else:
            return False
        
    def check_bounds(self, agent):
        # Check if a player has stepped out of bounds
        min_bounds = 1, 1 #min_width, min_length (zero-indexed)
        max_bounds = self.width - 2, self.length - 2
        agent.oob = ((agent.location <= min_bounds).any() or (agent.location >= max_bounds).any())

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

        # Draw the Agents last to ensure they always appear
        for agent in self.world.agents:
            pygame.draw.circle(
                canvas,
                self.colormap[agent.name],
                (agent.location + 0.5) * pix_square_size,
                pix_square_size / 3,
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
    
    def action_space(self, agent):
        return self.action_spaces[agent.name]
    
    def load_roster(self, file="Roster.csv"):
        roster = pd.read_csv(file)
        print(roster)
        for i, agent in enumerate(self.world.agents):
            agent.position = roster["Position"][i]
            agent.name = roster["Name"][i]

#TODO: Should probably either collapse this into the environment or refactor in a more logical way
class Scenario():
    def make_world(self, N=2):
        world = World()
        num_agents = N
        world.num_agents = num_agents
        num_defense = N/2
        num_landmarks = 1
        positions = ["WR", "DB"]
        world.timestep = None
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.defense = False if i < num_defense else True
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
    
    def reset_world(self, world):
        world.timestep = 0
        # Initial positions for landmark, agents
        goal = np.random.choice(world.landmarks)
        # Can build formations as an argument here
        starting_locations = {"WR_0":np.array([20, 10]), "DB_0":np.array([30,16])}
        routes = {"slant/in":np.array([30,20]), "go":np.array([50,10])}
        for agent in world.agents:
            agent.goal_a = goal
            agent.location = starting_locations[agent.name]
        for landmark in world.landmarks:
            # Can use landmarks to design plays for agents
            # Test randomly sampling which route to run (slant/in or go)
            landmark.location = routes["slant/in"]

    def agent_reward(self, agent, world):
        # Reward WR by how close they are to landmark and how far DB is from them
        # Currently doing well = positive reward values
        if agent.oob:
            return -100 #Large pentalty for stepping out of bounds
        elif np.sum(np.square(agent.location - agent.goal_a.location)) == 0:
            # If agent reaches target, give them a big reward
            return 50
        else:
            # Scale each component of the agent reward separately
            # Ex. 10 times more important to reach goal than to avoid CB
            defensive_players = self.defensive_players(world)
            def_rew = 0.0001 * sum(np.sqrt(np.sum(np.square(a.location - agent.location)))
                            for a in defensive_players)
            off_rew = -0.01 * np.sqrt(np.sum(np.square(agent.location - agent.goal_a.location)))
            time_penalty = -((world.timestep/10)**2) #Average NFL play lasts ~5s, motivate WR to get to target quickly
            return off_rew + def_rew + time_penalty
    
    def adversary_reward(self, agent, world):
        offensive_players = self.offensive_players(world)
        def_rew = -sum(np.sqrt(np.sum(np.square(agent.location - a.location)))
                      for a in offensive_players)
        return def_rew
    
    def reward(self, agent, world):
        return (
            self.adversary_reward(agent, world)
            if agent.defense
            else self.agent_reward(agent, world)
        )

    def defensive_players(self, world):
        return [agent for agent in world.agents if agent.defense]
    
    def offensive_players(self, world):
        return [agent for agent in world.agents if not agent.defense]