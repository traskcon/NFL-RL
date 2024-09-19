import numpy as np
from environments.envs import multiagent_environment

class raw_env(multiagent_environment):
    def __init__(self,
                 N=2,
                 max_cycles=25,
                 render_mode=None,
                ):
        scenario = Scenario()
        world = scenario.make_world(N)
        multiagent_environment.__init__(self,
                                        scenario=scenario,
                                        world=world,
                                        render_mode=render_mode,
                                        max_cycles=max_cycles)
        self.metadata["name"] = "db_battle-v0"

class Scenario():
    def make_world(self, N=2):
        world = World()
        num_agents = N
        world.num_agents = num_agents
        num_defense = N/2
        num_landmarks = 1
        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.defense = True if i < num_defense else False
            base_name = "defense" if agent.defense else "offense"
            base_index = i if i < num_defense else i - num_defense
            agent.name = f"{base_name}_{base_index}"
        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world
    
    def reset_world(self, world):
        # Initial positions for landmark, agents
        goal = np.random.choice(world.landmarks)
        for agent in world.agents:
            agent.goal_a = goal
            # Randomize starting locations for right now
            agent.location = np.random.randint(15,45,2)
        for landmark in world.landmarks:
            landmark.location = np.random.randint(15,45,2)

    def offensive_players(self, world):
        return [agent for agent in world.agents if not agent.defense]
    
    def defensive_players(self, world):
        return [agent for agent in world.agents if agent.defense]
    
    def reward(self, agent, world):
        return (
            self.defense_reward(agent, world)
            if agent.defense
            else self.agent_reward(agent, world)
        )
    
    def agent_reward(self, agent, world):
        # Reward WR by how close they are to landmark and how far DB is from them
        # Currently doing well = positive reward values
        defensive_players = self.defensive_players(world)
        def_rew = -sum(np.sqrt(np.sum(np.square(a.location - agent.location)))
                      for a in defensive_players)
        off_rew = np.sqrt(np.sum(np.square(agent.location - agent.goal_a.location)))
        return off_rew + def_rew
    
    def defense_reward(self, agent, world):
        # Reward DBs by how close they are to the receiver
        offensive_players = self.offensive_players(world)
        return -sum(np.sqrt(np.sum(np.square(agent.location - a.location)))
                    for a in offensive_players)
    
    def observation(self, agent, world):
        # Get positions of all other entities
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.location - agent.location)
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.location - agent.location)
        if not agent.defense:
            return np.concatenate([agent.goal_a.location - agent.state.location] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)

class Entity():
    def __init__(self):
        self.name = ""
        self.movable = False
        self.collide = True
        self.color = None
        self.location = None

class Agent(Entity):
    def __init__(self):
       super().__init__()
       self.movable = True 

class World():
    def __init__(self):
        self.agents = []
        self.landmarks = []
        #Future parameters added as needed to support continuous movement

class Landmark(Entity):
    def __init__(self):
        super().__init__()

db_battle_env = raw_env()