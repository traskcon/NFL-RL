import numpy as np
import pandas as pd
from environments.utils.core import Agent, World

class Scenario():
    def make_world(self, N=2, roster="Roster.csv"):
        world = World()
        num_agents = N
        world.num_agents = num_agents
        num_defense = N/2
        world.timestep = None
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        self.load_roster(world, roster)
        for i, agent in enumerate(world.agents):
            agent.defense = False if i < num_defense else True
            base_index = i if i < num_defense else int(i - num_defense)
            agent.name = f"{agent.position}_{base_index}"
        return world
    
    def reset_world(self, world):
        world.timestep = 0
        # Initial positions for landmark, agents
        # Can build formations as an argument here
        starting_locations = {"WR_0":np.array([20, 10]), "DB_0":np.array([30,16])}
        routes = {"slant/in":np.array([30,20]), "go":np.array([50,10]), "post":np.array([50,25])}
        for agent in world.agents:
            agent.location = starting_locations[agent.name]
            agent.target_location = routes["post"]

    def agent_reward(self, agent, world):
        # Reward WR by how close they are to landmark and how far DB is from them
        # Currently doing well = positive reward values
        if agent.oob:
            return -100 #Large pentalty for stepping out of bounds
        elif np.sum(np.square(agent.location - agent.target_location)) == 0:
            # If agent reaches target, give them a big reward
            return 50
        else:
            # Scale each component of the agent reward separately
            # Ex. 10 times more important to reach goal than to avoid CB
            defensive_players = self.defensive_players(world)
            def_rew = 0.0001 * sum(np.sqrt(np.sum(np.square(a.location - agent.location)))
                            for a in defensive_players)
            off_rew = -0.01 * np.sqrt(np.sum(np.square(agent.location - agent.target_location)))
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
    
    def load_roster(self, world, file="Roster.csv"):
        roster = pd.read_csv(file)
        for i, agent in enumerate(world.agents):
            agent.position = roster["position"][i]
            agent.strength = roster["strength"][i]
            agent.team = roster["team"][i]

    def update_agent_states(self):
        pass