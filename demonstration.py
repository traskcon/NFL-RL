# Script to demonstrate trained agents operating in the environment
from environments.envs import multiagent_environment
from environments import db_battle, all_22
import numpy as np
import policy

scenario = all_22.Scenario()
env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles=100, render_mode="human", roster="FullTeam-Roster.csv")
observations, infos = env.reset()
#learner = policy.Policy(env, observations)
#learner.load_models("-MK3_MR")

epsilon = 0.01

while env.world.agents:
    '''if np.random.random() <= epsilon:
        actions = {agent.name: env.action_space(agent).sample() for agent in env.world.agents}
    else:
        actions = {agent.name: learner.choose_action(agent, observations[agent.name], method="DQN")
               for agent in env.world.agents}
    new_observations, rewards, terminations, truncations, infos = env.step(actions)'''
    env.render()
    #observations = new_observations
env.close()
