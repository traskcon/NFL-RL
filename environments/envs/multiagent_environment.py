import pandas as pd
import numpy as np
from gymnasium import spaces
import pygame
from environments.utils.core import Agent, World
from pettingzoo import ParallelEnv

class MultiEnvironment(ParallelEnv):
    metadata = {"name":"multiagent_environment-v0", "render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, max_cycles, scenario, render_mode=None, width=124, length=57, roster="Roster.csv"):
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
        self.roster = roster
        self.target_location = np.array([None, None])
        self.timestep = None
        self.world = scenario.make_world(roster=roster)
        self.agent_names = [agent.name for agent in self.world.agents]
        self.colormap = {"Lions":(0,118,182), "Broncos":(255,82,0), "Commanders":(90,20,20),
                         "Cowboys":(0,34,68), "Cardinals":(135,0,39), "Patriots":(0,21,50),
                         "49ers":(170,0,0), "Vikings":(79,38,131), "Bills":(12,46,130),
                         "Falcons":(163,13,45), "Ravens":(26,25,95), "Panthers":(0,133,202),
                         "Bears":(200,65,0), "Bengals":(211,47,30), "Browns":(49,29,0),
                         "Packers":(28,45,37), "Texans":(0,7,28), "Colts":(1,51,105),
                         "Jaguars":(0,101,118), "Chiefs":(227,24,55), "Chargers":(238,173,30),
                         "Rams":(0,53,148), "Dolphins":(0,142,151), "Saints":(159,137,88),
                         "Giants":(155,161,162), "Jets":(0,63,45), "Raiders":(0,0,0),
                         "Eagles":(0,76,84), "Steelers":(238,173,30), "Seahawks":(105,190,40),
                         "Buccaneers":(213,10,10), "Titans":(68,149,210)}

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
        self.world = self.scenario.make_world(roster=self.roster)
        self.scenario.reset_world(self.world)

        self.rewards = {name: 0.0 for name in self.agent_names}
        self.terminations = {name: False for name in self.agent_names}
        self.target_location = self.world.agents[0].target_location

        observations = {a.name:(
            *[agent.location for agent in self.world.agents],
            a.target_location,
        ) for a in self.world.agents}

        if self.render_mode == "human":
            self._render_frame()

        return observations

    def step(self, actions):
        '''Set action for all agents, update relevant states'''
        for agent in self.world.agents:
            agent_action = actions[agent.name] # Execute action
            agent_direction = self._action_to_direction[agent_action] #Map the action (0,1,2,3) to the direction of movement
            player_locations = np.array([agent.location for agent in self.world.agents])
            collisions = np.all((agent.location + agent_direction) == player_locations, axis=1)
            #TODO: Repackage collision checking into its own function
            # Also need to investigate whether or not 
            if np.any(collisions):
                i = np.flatnonzero(collisions)[0]
                opp_agent = self.world.agents[i]
                strength_diff = agent.strength - opp_agent.strength
                #Sampling from normal distribution
                if strength_diff >= np.random.randn()*10:
                    # Print option for debugging
                    # print("{} pushed {} backwards".format(agent.name, opp_agent.name))
                    # Push opponent backwards
                    opp_agent.location = opp_agent.location + agent_direction
                    # Use np.clip to ensure we don't leave the grid
                    agent.location = np.clip(
                        agent.location + agent_direction, [0, 0], [self.width - 1, self.length - 1]
                    )
                # Otherwise don't move
            else:
                agent.location = np.clip(
                    agent.location + agent_direction, [0, 0], [self.width - 1, self.length - 1]
                )

        # Update each agent's position-specific states (passing vs scrambling, ballcarrier)
        self.scenario.update_agent_states(self.world)

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
            a.target_location - a.location,
        ) for a in self.world.agents}

        if any(self.terminations.values()) or all(truncations.values()):
            # If termination/truncation condition met, remove all agents
            self.world.agents = []
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observations, self.rewards, self.terminations, truncations
    
    def termination(self, agent):
        # First check for scenario-specific terminations
        if self.scenario.termination(agent, self.world):
            return True
        else:
            # Check if an offensive player has stepped out of bounds
            if not agent.defense:
                self.check_bounds(agent)
                return agent.oob
            else:
                return False
        
    def check_bounds(self, agent):
        # Check if a player has stepped out of bounds by seeing if they've left the field of play
        field_of_play = np.array([[2,2],
                                  [self.width-2, self.length-2]])
        agent.oob = not self.check_in_box(field_of_play, agent.location) 

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
        # Draw gridlines
        for x in range(self.length + 1):
            pygame.draw.line(
                canvas,
                (0,0,0, 0.2),
                (0, pix_square_size * x),
                (self.window_width, pix_square_size * x),
                width=1,
            )
        for y in range(self.width + 1):
            if y == self.world.yardline:
                pygame.draw.line(
                    canvas,
                    (0,0,255, 1),
                    (pix_square_size * y, 0),
                    (pix_square_size * y, self.window_length),
                    width=1,
                )
            else:
                pygame.draw.line(
                    canvas,
                    (0,0,0, 0.2),
                    (pix_square_size * y, 0),
                    (pix_square_size * y, self.window_length),
                    width=1,
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
                self.colormap[agent.team],
                (agent.location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            if agent.ballcarrier:
                pygame.draw.circle(
                    canvas,
                    (255,255,0),
                    (agent.location + 0.5) * pix_square_size,
                    pix_square_size/2, width=1
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
    
    def check_in_box(self, bounding_box, point):
        # Given a 2D point and a bounding box defined by [[x1, y1], [x2, y2]], return whether the point is within the box
        return ((point>=bounding_box[0]) & (point<=bounding_box[1])).all()