# Load environment
from environments.envs import multiagent_environment

scenario = multiagent_environment.Scenario()
env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles = 100, render_mode="human")
observations, infos = env.reset()

while env.world.agents:
    # this is where you would insert your policy
    actions = {agent.name: env.action_space(agent).sample() for agent in env.world.agents}
    actions["WR_0"] = 2
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(terminations)
    env.render()

env.close()
