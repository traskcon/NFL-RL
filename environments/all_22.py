import numpy as np
from environments.utils.core import Agent, World
import pandas as pd

class Scenario():
    def make_world(self, N=22, roster="Roster.csv"):
        world = World()
        num_agents = N
        world.num_agents = num_agents
        world.timestep = None
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        self.load_roster(world, roster)
        return world
    
    def reset_world(self, world):
        # Will need multiple reset functions
        # One to reset to start of the play, one for start of the drive, and one for start of game
        world.timestep = 0
        # Can build formations as an argument here
        self.load_play(world)
        self.yardline = 30
        # Define the pocket as a 6-yard wide box around QB's initial position
        qb = [player for player in world.agents if player.position == "QB"][0]
        self.pocket = np.array([[qb.location[0]-3, qb.location[1]+3],
                                [qb.location[0]+3, qb.location[1]-3]])
        
    def update_agent_states(self, world):
        for player in world.agents:
            if player.position == "QB":
                # If QB is in the pocket, set their status as passing
                player.passing = self.check_in_box(self.pocket, player.location)
            # Add catching, handoff, and fumble functions here to change ballcarrier
        
    def check_in_box(self, bounding_box, point):
        # Given a 2D point and a bounding box defined by [[x1, y1], [x2, y2]], return whether the point is within the box
        return ((point>=bounding_box[0]) & (point<=bounding_box[1])).all(1)

    def qb_reward(self, agent, world):
        #QB has a two-level reward probably
        #Encode two states for QB: Passing & Scrambling
        if agent.passing:
            # If QB is in passing mode, reward them for staying away from DL
            # May need to add reward to encourage QB to stay in the pocket
            dl_players = [player for player in self.defensive_players(world) if player.position == "DL"]
            ol_rew = sum(np.sqrt(np.sum(np.square(a.location - agent.location)))
                        for a in dl_players)
            return ol_rew
        else:
            # Otherwise reward them for getting downfield, avoiding nearest defender
            downfield_rew = agent.location[0] - self.yardline
            defense = [player.location for player in self.defensive_players(world)]
            evasion_rew = np.min([np.sqrt(np.sum(np.square(agent.location - loc))) for loc in defense])
            return downfield_rew + evasion_rew

    def rb_reward(self, agent, world):
        # Currently identical to WR reward
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
            off_rew = -0.01 * np.sqrt(np.sum(np.square(agent.location - agent.target_location)))
            time_penalty = -((world.timestep/10)**2) #Average NFL play lasts ~5s, motivate WR to get to target quickly
            return off_rew + def_rew + time_penalty

    def te_reward(self, agent, world):
        # TE reward depends on playcall (Blocking or route-running)
        pass

    def ol_reward(self, agent, world):
        # reward = sum(dist(DL, QB))
        dl_players = [player for player in self.defensive_players(world) if player.position == "DL"]
        qb = [player for player in world.agents if player.position == "QB"][0]
        ol_rew = sum(np.sqrt(np.sum(np.square(a.location - qb.location)))
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
            off_rew = -0.01 * np.sqrt(np.sum(np.square(agent.location - agent.target_location)))
            time_penalty = -((world.timestep/10)**2) #Average NFL play lasts ~5s, motivate WR to get to target quickly
            return off_rew + def_rew + time_penalty
    
    def db_reward(self, agent, world):
        #Reward is distance from closest WR
        receivers = [player.location for player in self.offensive_players(world) if player.position == "WR"]
        db_rew = -1*np.min([np.sqrt(np.sum(np.square(agent.location - loc))) for loc in receivers])
        return db_rew
    
    def lb_reward(self, agent, world):
        #TODO: Determine a reward function for LB
        pass
    
    def dl_reward(self, agent, world):
        # reward = -dist(agent, QB)
        qb = [player for player in world.agents if player.position == "QB"][0]
        dl_rew = -np.sqrt(np.sum(np.square(agent.location - qb.location)))
        return dl_rew

    def reward(self, agent, world):
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

    def load_play(self, world, file="Test-Playbook.csv"):
        # CSV file containing player starting locations for different plays
        #routes = {"slant/in":np.array([30,20]), "go":np.array([50,10])}
        #landmark.location = routes["slant/in"]
        # TODO: Add flag indicating whether a player is eligible or not, use that for coverage rewards
        play = pd.read_csv(file)
        for i, agent in enumerate(world.agents):
            row = play.index[play["position"] == agent.index].to_list()[0]
            agent.location = np.array([play.at[row,"x"],play.at[row,"y"]])
            agent.target_location = np.array([play.at[row,"goal_x"], play.at[row,"goal_y"]])