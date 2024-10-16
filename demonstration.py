# Script to demonstrate trained agents operating in the environment
from environments.envs import multiagent_environment
import numpy as np
import policy

scenario = multiagent_environment.Scenario()
env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles=100, render_mode="human")
observations, infos = env.reset()
learner = policy.Policy(env, observations)
learner.load_models("BL")

while env.world.agents:
    actions = {agent.name: learner.choose_action(agent, observations[agent.name], method="dqn")
               for agent in env.world.agents}
    new_observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()
    observations = new_observations
env.close()
