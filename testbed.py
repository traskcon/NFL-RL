# Load environment
from environments.envs import multiagent_environment
import numpy as np
import policy
from collections import deque
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

scenario = multiagent_environment.Scenario()
env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles = 100, render_mode=None)
observations, infos = env.reset()
learner = policy.Policy(env, observations)

train_episodes = 300
epsilon = 1 #Epsilon-greedy algorithm, every step is random initially
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

replay_memory = deque(maxlen=50_000)
cumulative_rewards = {agent.name: [] for agent in env.world.agents}

for episode in tqdm(range(train_episodes)):
    observations, infos = env.reset()
    for agent in env.world.agents:
        cumulative_rewards[agent.name].append(0)
    while env.world.agents:
        if np.random.rand() <= epsilon:
            actions = {agent.name: env.action_space(agent).sample() for agent in env.world.agents}
        else:
            actions = {agent.name: learner.choose_action(agent, observations[agent.name], method="dqn") 
                       for agent in env.world.agents}
        new_observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent in env.world.agents:
            cumulative_rewards[agent.name][-1] += rewards[agent.name]
        replay_memory.append([observations, actions, rewards, new_observations, terminations])
        if episode % 10 == 0:
            # Network training is currently quite time expensive, explore training less frequently
            [learner.train(replay_memory, name) for name in env.agent_names]
        env.render()
        observations = new_observations
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    # Visualize the final trained agents
    if episode == (train_episodes - 2):
        env.render_mode = "human"
env.close()

#Visualize cumulative rewards
for agent, rewards in cumulative_rewards.items():
    plt.plot(range(train_episodes), rewards, label=agent)
plt.legend()
plt.show()