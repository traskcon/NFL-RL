# Load environment
from environments.envs import multiagent_environment
import policy

scenario = multiagent_environment.Scenario()
env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles = 100, render_mode="human")
observations, infos = env.reset()
learner = policy.Policy(env)

while env.world.agents:
    # this is where you would insert your policy
    actions = {agent.name: learner.choose_action(agent, observations) for agent in env.world.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()

env.close()
