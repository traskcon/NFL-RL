# Load environment
from environments.envs import multiagent_environment
from environments import db_battle, all_22
import numpy as np
import policy
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

scenario = all_22.Scenario()
env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles = 100, render_mode=None, roster="FullTeam-Roster.csv")
observations = env.reset()
learner = policy.Policy(env, observations)

train_episodes = 300
epsilon = 1 #Epsilon-greedy algorithm, every step is random initially
max_epsilon = 1
min_epsilon = 0.01
decay = 0.005
world_steps = 0
consecutive_terminations = 0

replay_memory = deque(maxlen=50_000)
cumulative_rewards = {agent.name: [] for agent in env.world.agents}
q_loss = {agent.name: [] for agent in env.world.agents}

for episode in tqdm(range(train_episodes)):
    observations = env.reset()
    for agent in env.world.agents:
        cumulative_rewards[agent.name].append(0)
    while env.world.agents:
        world_steps += 1
        decision = np.random.rand()
        if decision <= epsilon:
            actions = {agent.name: env.action_space(agent).sample() for agent in env.world.agents}
        else:
            actions = {agent.name: learner.choose_action(agent, observations[agent.name], method="dqn") 
                       for agent in env.world.agents}
        new_observations, rewards, terminations, truncations = env.step(actions)
        for agent in env.world.agents:
            cumulative_rewards[agent.name][-1] += rewards[agent.name]
        replay_memory.append([observations, actions, rewards, new_observations, terminations])
        if (world_steps % 10 == 0) or any(terminations.values()) or all(truncations.values()):
            # Train Q-Networks every 5 simulation steps or at simulation end
            [learner.train(replay_memory, name) for name in env.agent_names]
            try:
                [q_loss[name].append(learner.history[name].history["loss"]) for name in env.agent_names]
            except:
                pass
        if world_steps >= 100:
            learner.copy_weights()
            world_steps = 0
        env.render()
        observations = new_observations
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    # Visualize the final trained agents
    if episode == (train_episodes - 2):
        env.render_mode = "human"
env.close()
learner.save_models("-A22-MK1")

#Visualize cumulative rewards
i = 1
for agent, rewards in cumulative_rewards.items():
    plt.subplot(5,5,i)
    plt.gca().set_title(agent + " Reward")
    plt.plot(range(len(rewards)), rewards)
    i += 1
for agent, loss in q_loss.items():
    plt.subplot(5,5,i)
    plt.gca().set_title(agent + " Loss")
    plt.plot(range(len(loss)), loss)
    i += 1
plt.show()