import numpy as np
import copy

class Policy():
    # NOTE: Each agent should have its own policy, observations in order to minimize Q_table size 
    # for full simulation. Initial example uses a common q_table for all agents
    def __init__(self, env):
        # Environment state is a tuple of n 1x2 np arrays containing each agent's position
        self.env = env
        self.reward_function = env.scenario.reward

    def choose_action(self, agent, observations):
        # Environment state is a tuple of n 1x2 np arrays containing each agent's position
        # Initial basic decision-making policy:
        # Choose action that gives the largest reward based on the current state
        rewards = dict()
        temp_agent = copy.copy(agent)
        for action, direction in self.env._action_to_direction.items():
            temp_agent.location = agent.location + direction
            rewards[action] = self.reward_function(temp_agent, self.env.world)
        return max(rewards, key = rewards.get)
        
