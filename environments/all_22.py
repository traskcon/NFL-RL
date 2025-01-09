import numpy as np
from environments.utils.core import Agent, World
import pandas as pd
import sys
import json

class Scenario():
    def make_world(self, N=22, roster="Roster.csv"):
        world = World()
        num_agents = N
        world.num_agents = num_agents
        world.timestep = None
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        self.load_roster(world, roster)
        self.load_playbook(world)
        return world
    
    def reset_world(self, world):
        # Will need multiple reset functions
        # One to reset to start of the play, one for start of the drive, and one for start of game
        world.timestep = 0
        # Can build formations as an argument here
        self.yardline = 30
        world.yardline = 30
        self.load_play(world)
        self.active_endzone = np.array([[112,2],
                                        [122,55]]) #Hard-coded rn, fix in future to vary with environment
        # Define the pocket as a 6-yard wide box around QB's initial position
        # Index the QB, DL once at game start instead of everytime reward function is run
        self.qb_index = self.get_player_indices(world, ["QB"])[0]
        self.dl_indices = self.get_player_indices(world, ["DL"])
        self.wr_indices = self.get_player_indices(world, ["WR"])
        qb = world.agents[self.qb_index]
        self.pocket = np.array([qb.location-3, qb.location+3])
        qb.ballcarrier = True #Define QB as ballcarrier at start of the play
        self.update_agent_states(world)

    def get_player_indices(self, world, position):
        # Return a list of player indices from world.agents based on player position
        return [index for index, player in enumerate(world.agents) if player.position in position]
        
    def update_agent_states(self, world):
        qb = world.agents[self.qb_index]
        # If QB is in the pocket, set their status as passing
        qb.passing = self.check_in_box(self.pocket, qb.location)
        # Need to determine how QB decides who to pass to
        # Add catching, handoff, and fumble functions here to change ballcarrier

    def termination(self, agent, world, verbose=False): 
        if agent.ballcarrier:
            if self.check_in_box(self.active_endzone, agent.location):
                return True #End if touchdown is scored
            else:
                # Check if any defenders are adjacent
                tackle_box = np.array([agent.location-1, agent.location+1])
                tacklers = [defense for defense in self.defensive_players(world) if self.check_in_box(tackle_box, defense.location)]
                for tackler in tacklers:
                    tackle_diff = tackler.tackle - agent.break_tackle
                    if tackle_diff >= np.random.randn()*10:
                        #Sampling from normal distribution, potentially readjust based on sim results
                        if verbose: print("{} tackled {} at the {} yardline".format(tackler.name, agent.name, agent.location[0]))
                        return True
        else:
            return False
        
    def check_in_box(self, bounding_box, point):
        # Given a 2D point and a bounding box defined by [[x1, y1], [x2, y2]], return whether the point is within the box
        return ((point>=bounding_box[0]) & (point<=bounding_box[1])).all()

    def qb_reward(self, agent, world):
        #QB has a two-level reward probably
        #Encode two states for QB: Passing & Scrambling
        if agent.passing:
            # If QB is in passing mode, reward them for staying away from DL
            # May need to add reward to encourage QB to stay in the pocket
            dl_players = [world.agents[i] for i in self.dl_indices]
            ol_rew = sum(np.sqrt(np.sum(np.square(a.location - agent.location)))
                        for a in dl_players)
            return ol_rew
        else:
            # Otherwise reward them for getting downfield, avoiding nearest defender
            downfield_rew = agent.location[0] - self.yardline
            defense = [player.location for player in self.defensive_players(world)]
            evasion_rew = np.min([np.sqrt(np.sum(np.square(agent.location - loc))) for loc in defense])
            return downfield_rew + 0.1 * evasion_rew

    def rb_reward(self, agent, world):
        # Currently identical to WR reward
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

    def te_reward(self, agent, world):
        # TE reward depends on playcall (Blocking or route-running)
        # For now just use OL reward and assume blocking
        dl_players = [world.agents[i] for i in self.dl_indices]
        qb = [player for player in world.agents if player.position == "QB"][0]
        ol_rew = sum(np.sqrt(np.sum(np.square(a.location - qb.location)))
                      for a in dl_players)
        return ol_rew

    def ol_reward(self, agent, world):
        # Intiial reward: reward = sum(dist(DL, QB))
        # Reworked reward: reward = min(dist(DL, OL)) + sum(dist(DL, QB))
        # Pocket breaks down very quickly, likely need to change movement speed to prevent that
        dl_players = [world.agents[i] for i in self.dl_indices]
        qb = world.agents[self.qb_index]
        pass_pro = sum(np.sqrt(np.sum(np.square(a.location - qb.location)))
                      for a in dl_players)
        run_block = -1*np.min([np.sqrt(np.sum(np.square(agent.location - dl.location))) for dl in dl_players])
        return pass_pro + run_block

    def wr_reward(self, agent, world):
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
    
    def db_reward(self, agent, world):
        #Reward is distance from closest WR
        receivers = [world.agents[i] for i in self.wr_indices]
        db_rew = -1*np.min([np.sqrt(np.sum(np.square(agent.location - loc.location))) for loc in receivers])
        return db_rew
    
    def lb_reward(self, agent, world):
        #Using a modified WR reward function as an approximator for zone coverage
        # Cover anybody who comes close to zone landmark
        receivers = [player.location for player in self.offensive_players(world)]
        # Define zone as 10x10 box centered on target location
        zone = np.array([agent.target_location-5, agent.target_location+5])
        coverage_rew = -1*sum([np.sqrt(np.sum(np.square(agent.location - loc))) for loc in receivers if self.check_in_box(zone, loc)])
        zone_rew = -0.01 * np.sqrt(np.sum(np.square(agent.location - agent.target_location)))
        return coverage_rew + zone_rew
    
    def dl_reward(self, agent, world):
        # reward = -dist(agent, QB)
        qb = world.agents[self.qb_index]
        dl_rew = -np.sqrt(np.sum(np.square(agent.location - qb.location)))
        return dl_rew
    
    def bc_reward(self, agent, world):
        # Reward function for ballcarriers, incentivize running towards endzone, away from defense
        forward_progress = agent.location[0] - self.yardline
        def_loc = [player.location for player in self.defensive_players(world)]
        evasion = np.min([np.sqrt(np.sum(np.square(agent.location - loc))) for loc in def_loc])
        return forward_progress + evasion
    
    def pursuit_reward(self, agent, world):
        # Reward function for defense chasing the ballcarrier
        ballcarrier = [player for player in world.agents if player.ballcarrier][0]
        pursuit = -1*np.sqrt(np.sum(np.square(agent.location - ballcarrier.location)))
        return pursuit

    def reward(self, agent, world):
        #Position-specific reward given at each timestep during a play (reward-shaping)
        ballcarrier = [player for player in world.agents if player.ballcarrier][0]
        if (ballcarrier.location[0] > self.yardline) and (agent.defense or agent.ballcarrier):
            # If ballcarrier is past the line of scrimmage, return special ballcarrier and defense rewards
            if agent.defense:
                return self.pursuit_reward(agent, world)
            elif agent.ballcarrier:
                return self.bc_reward(agent, world)
        else:
            # Otherwise return standard position rewards
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
        # HOLD FOR FUTURE PLAYCALLING AI
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
            agent.tackle = roster["tackle"][i]
            agent.break_tackle = roster["break_tackle"][i]
            agent.defense = True if agent.position in ["DB","DL","LB"] else False
            position_counts[agent.position] = position_counts.get(agent.position, 0) + 1
            agent.index = f"{agent.position}_{position_counts[agent.position]}"

    def load_playbook(self, world, file="sample_playbook.json"):
        #Each team has a JSON file containing all of their plays and formations
        #These plays can then be sampled from and loaded in for each down
        with open(file) as json_data:
            self.playbook = json.load(json_data)

    def load_play(self, world, file="Test-Playbook.csv"):
        # CSV file containing player starting locations for different plays
        #routes = {"slant/in":np.array([30,20]), "go":np.array([50,10])}
        #landmark.location = routes["slant/in"]
        # TODO: Add flag indicating whether a player is eligible or not, use that for coverage rewards
        filetype = file.split(".")[-1].lower()
        if filetype == "csv":
            play = pd.read_csv(file)
            for i, agent in enumerate(world.agents):
                row = play.index[play["position"] == agent.index].to_list()[0]
                agent.location = np.array([play.at[row,"x"],play.at[row,"y"]])
                agent.target_location = np.array([play.at[row,"goal_x"], play.at[row,"goal_y"]])
        elif filetype == "json":
            # TEMP PLACEHOLDER: Update with correct parser using relative formation locations
            with open(file) as json_data:
                play = json.load(json_data)
                for agent in world.agents:
                    agent.location = np.array(play["formation"][agent.index])
                    agent.location[0] += self.yardline
                    agent.target_location = np.array(play["target_locations"][agent.index])
        else:
            sys.exit("Attempted to load invalid play filetype")