import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import random
import keras
import copy

class Policy():
    # NOTE: Each agent should have its own policy, observations in order to minimize Q_table size 
    # for full simulation. Initial example uses a common q_table for all agents
    def __init__(self, env, observations):
        # Environment state is a tuple of n 1x2 np arrays containing each agent's position
        self.env = env
        self.reward_function = env.scenario.reward
        self.models = dict()
        for agent, state in observations.items():
            self.models[agent] = self.build_dqn(state)

    def choose_action(self, agent, observation=None, method="short-term"):
        # Environment state is a tuple of n 1x2 np arrays containing each agent's position
        # Initial basic decision-making policy:
        # Choose action that gives the largest reward based on the current state
        if method == "dqn":
            model = self.models[agent.name]
            observation = np.array(observation)
            observation_reshaped = np.reshape(observation, [1, *observation.shape])
            predicted = model.predict(observation_reshaped, verbose=0).flatten()
            return np.argmax(predicted) # Return action with predicted highest Q-value
        else:
            rewards = dict()
            temp_agent = copy.copy(agent)
            for action, direction in self.env._action_to_direction.items():
                temp_agent.location = agent.location + direction
                rewards[action] = self.reward_function(temp_agent, self.env.world)
            return max(rewards, key = rewards.get)
    
    def build_dqn(self, state):
        learning_rate = 0.001
        action_shape = 4
        state = np.array(state)
        init = keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Input(state.shape))
        model.add(keras.layers.Dense(24, activation="relu",kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation="relu", kernel_initializer=init))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(action_shape, activation="linear", kernel_initializer=init))
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])
        return model
    
    def train(self, replay_memory, agent):
        model = self.models[agent]
        learning_rate = 0.7
        discount_factor = 0.618

        MIN_REPLAY_SIZE = 1000
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return
        
        batch_size = 128
        mini_batch = random.sample(replay_memory, batch_size)
        current_states = np.array([info[0][agent] for info in mini_batch])
        current_qs_list = model.predict(current_states, verbose=0)
        next_states = np.array([info[3][agent] for info in mini_batch])
        future_qs_list = model.predict(next_states, verbose=0)

        X = []
        Y = []
        for i, (observation, actions, reward, new_observation, terminated) in enumerate(mini_batch):
            action = actions[agent]
            if not terminated:
                max_future_q = reward[agent] + discount_factor * np.max(future_qs_list[i])
            else:
                max_future_q = reward[agent]

            current_qs = current_qs_list[i,:]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(observation[agent])
            Y.append(current_qs)
        model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

