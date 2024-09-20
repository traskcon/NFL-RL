import gymnasium as gym
# Load environment
from environments.envs import multiagent_environment

env = multiagent_environment.MultiEnvironment(max_cycles = 100, render_mode="human")
observations, infos = env.reset()

while env.world.agents:
    # this is where you would insert your policy
    actions = {agent.name: env.action_space(agent).sample() for agent in env.world.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()

env.close()
