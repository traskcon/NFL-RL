# Load environment
from environments.envs import multiagent_environment
import policy

scenario = multiagent_environment.Scenario()
env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles = 100, render_mode="human")
observations, infos = env.reset()
learner = policy.Policy(env, observations)

train_episodes = 300

for episode in range(train_episodes):
    observations, infos = env.reset()
    
    while env.world.agents:
        # this is where you would insert your policy
        actions = {agent.name: learner.choose_action(agent) for agent in env.world.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        #env.render()

env.close()
