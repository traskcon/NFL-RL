from environments.envs import multiagent_environment
from environments import all_22
import policy
import numpy as np

scenario = all_22.Scenario()
env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles = 100, render_mode="human", roster="FullTeam-Roster.csv")
observations = env.reset()
learner = policy.Policy(env, observations)
learner.load_models("-A22-MK1")

while env.world.down < 5:
    observations = env.reset()
    while env.world.agents:
        actions = {agent.name: learner.choose_action(agent, observations[agent.name], method="dqn")
            for agent in env.world.agents}
        new_observations, rewards, terminations, truncations = env.step(actions)
        env.render()
        observations = new_observations
env.close()