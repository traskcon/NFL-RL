import numpy as np
from environments.utils.core import Agent, Landmark, World
import pandas as pd

class Scenario():
    def make_world(self, N=22, roster="Roster.csv"):
        world = World()
        num_agents = N
        world.num_agents = num_agents
        num_landmarks = 1
        world.timestep = None
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        self.load_roster(world, roster)
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
        self.load_play(world)
        self.yardline = 30

    def qb_reward(self, agent, world):
        #QB has a two-level reward probably
        pass

    def rb_reward(self, agent, world):
        # Probably just use WR reward
        pass

    def te_reward(self, agent, world):
        # TE reward depends on playcall (Blocking or route-running)
        pass

    def ol_reward(self, agent, world):
        # reward = sum(dist(DL, QB))
        dl_players = [player for player in self.defensive_players(world) if player.position == "DL"]
        qb = [player for player in world.agents if player.position == "QB"][0]
        ol_rew = -sum(np.sqrt(np.sum(np.square(a.location - qb.location)))
                      for a in dl_players)
        return ol_rew

    def wr_reward(self, agent, world):
        # Reward WR by how close they are to landmark and how far DB is from them
        # Currently doing well = positive reward values
        if agent.oob:
            return -100 #Large pentalty for stepping out of bounds
        elif np.sum(np.square(agent.location - agent.goal.location)) == 0:
            # If agent reaches target, give them a big reward
            return 50
        else:
            # Scale each component of the agent reward separately
            # Ex. 10 times more important to reach goal than to avoid CB
            defensive_players = self.defensive_players(world)
            def_rew = 0.0001 * sum(np.sqrt(np.sum(np.square(a.location - agent.location)))
                            for a in defensive_players)
            off_rew = -0.01 * np.sqrt(np.sum(np.square(agent.location - agent.goal.location)))
            time_penalty = -((world.timestep/10)**2) #Average NFL play lasts ~5s, motivate WR to get to target quickly
            return off_rew + def_rew + time_penalty
    
    def db_reward(self, agent, world):
        #Reward is distance from closest WR
        receivers = [player for player in self.offensive_players(world) if player.position == "WR"]
        db_rew = -np.min(np.sqrt(np.sum(np.square(agent.location - a.location)))
                      for a in receivers)
        return db_rew
    
    def lb_reward(self, agent, world):
        #TODO: Determine a reward function for LB
        pass
    
    def dl_reward(self, agent, world):
        # reward = -dist(agent, QB)
        pass

    def step_reward(self, agent, world):
        #Position-specific reward given at each timestep during a play (reward-shaping)
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
    
    def play_reward(self, agent, world):
        #Reward based on TD, turnover, EPA
        #Calculate EPA using ep_model.py
        pass

    def defensive_players(self, world):
        return [agent for agent in world.agents if agent.defense]
    
    def offensive_players(self, world):
        return [agent for agent in world.agents if not agent.defense]
    
    def load_roster(self, world, file="Roster.csv"):
        roster = pd.read_csv(file)
        position_counts = dict()
        for i, agent in enumerate(world.agents):
            agent.position = roster["position"][i]
            agent.name = roster["name"][i]
            agent.strength = roster["strength"][i]
            agent.team = roster["team"][i]
            agent.defense = True if agent.position in ["DB","DL","LB"] else False
            position_counts[agent.position] = position_counts.get(agent.position, 0) + 1
            agent.index = f"{agent.position}_{position_counts[agent.position]}"
            agent.location = np.array([None, None])
            agent.goal = Landmark()

    def load_play(self, world, file="Test-Playbook.csv"):
        # CSV file containing player starting locations for different plays
        #routes = {"slant/in":np.array([30,20]), "go":np.array([50,10])}
        #landmark.location = routes["slant/in"]
        # TODO: Add flag indicating whether a player is eligible or not, use that for coverage rewards
        play = pd.read_csv(file)
        for i, agent in enumerate(world.agents):
            row = play.index[play["position"] == agent.index].to_list()[0]
            agent.location = np.array([play.at[row,"x"],play.at[row,"y"]])
            agent.goal.location = np.array([play.at[row,"goal_x"], play.at[row,"goal_y"]])