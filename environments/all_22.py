import numpy as np
from environments.utils.core import Agent, Landmark, World
import pandas as pd

class Scenario():
    def make_world(self, N=22, roster="Roster.csv"):
        world = World()
        num_agents = N
        world.num_agents = num_agents
        num_defense = N/2
        num_landmarks = 1
        world.timestep = None
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        self.load_roster(roster)
        for i, agent in enumerate(world.agents):
            agent.defense = True if agent.position in ["DB","DL","LB"] else False
            base_index = i if i < num_defense else int(i - num_defense)
            agent.index = f"{agent.position}_{base_index}"
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
        self.load_play()
        starting_locations = {"WR_0":np.array([20, 10]), "DB_0":np.array([30,16])}
        routes = {"slant/in":np.array([30,20]), "go":np.array([50,10])}
        for agent in world.agents:
            agent.goal_a = goal
            agent.location = starting_locations[agent.name]
        for landmark in world.landmarks:
            # Can use landmarks to design plays for agents
            # Test randomly sampling which route to run (slant/in or go)
            # landmark.location = routes[np.random.choice(list(routes.keys()))]
            landmark.location = routes["slant/in"]

    def qb_reward(self, agent, world):
        pass

    def rb_reward(self, agent, world):
        pass

    def te_reward(self, agent, world):
        pass

    def ol_reward(self, agent, world):
        pass

    def wr_reward(self, agent, world):
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
    
    def db_reward(self, agent, world):
        offensive_players = self.offensive_players(world)
        def_rew = -sum(np.sqrt(np.sum(np.square(agent.location - a.location)))
                      for a in offensive_players)
        return def_rew
    
    def lb_reward(self, agent, world):
        pass
    
    def dl_reward(self, agent, world):
        pass

    def reward(self, agent, world):
        reward_dict = {
            "QB": self.qb_reward,
            "RB": self.rb_reward,
            "TE": self.te_reward,
            "OL": self.ol_reward,
            "WR": self.wr_reward,
            "DB": self.db_reward,
            "LB": self.lb_reward,
            "DL": self.dl_reward,
        }
        reward_func = reward_dict[agent.position]
        return reward_func(agent, world)

    def defensive_players(self, world):
        return [agent for agent in world.agents if agent.defense]
    
    def offensive_players(self, world):
        return [agent for agent in world.agents if not agent.defense]
    
    def load_roster(self, file="Roster.csv"):
        roster = pd.read_csv(file)
        for i, agent in enumerate(self.world.agents):
            agent.position = roster["Position"][i]
            agent.name = roster["Name"][i]

    def load_play(self, file="Playbook.csv"):
        # CSV file containing player starting locations for different plays
        pass