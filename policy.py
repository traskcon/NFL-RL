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
        self.q_models = dict() # Q-Network Models for each agent
        self.t_models = dict() # Target Network Models for each agent
        self.history = dict()
        for agent, state in observations.items():
            self.q_models[agent] = self.build_dqn(state)
            self.t_models[agent] = self.build_dqn(state)

    def choose_action(self, agent, observation=None, method="short-term"):
        # Environment state is a tuple of n 1x2 np arrays containing each agent's position
        if method == "dqn":
            model = self.q_models[agent.name]
            observation = np.array(observation)
            observation_reshaped = np.reshape(observation, [1, *observation.shape])
            predicted = model.predict(observation_reshaped, verbose=0).flatten()
            return np.argmax(predicted) # Return action with predicted highest Q-value
        else:
            # Heuristic algorithm: Take action with largest immediate reward
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
        ''' Train the Q-Networks '''
        q_model = self.q_models[agent]
        t_model = self.t_models[agent]
        # Explore adjusting learning rate, discount factor to shape behavior
        learning_rate = 0.7
        discount_factor = 0.9

        MIN_REPLAY_SIZE = 1000
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return
        
        batch_size = 128
        mini_batch = random.sample(replay_memory, batch_size)
        current_states = np.array([info[0][agent] for info in mini_batch])
        predicted_qs_list = q_model.predict(current_states, verbose=0)
        next_states = np.array([info[3][agent] for info in mini_batch])
        future_qs = t_model.predict(next_states, verbose=0)

        X = []
        Y = []
        for i, (observation, actions, reward, new_observation, terminated) in enumerate(mini_batch):
            action = actions[agent]
            if not terminated:
                max_future_q = reward[agent] + discount_factor * np.max(future_qs[i,:])
            else:
                max_future_q = reward[agent]

            predicted_qs = predicted_qs_list[i,:]
            predicted_qs[action] = (1 - learning_rate) * predicted_qs[action] + learning_rate * max_future_q

            X.append(observation[agent])
            Y.append(predicted_qs)
        self.history[agent] = q_model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    def copy_weights(self):
        for agent, q_model in self.q_models.items():
            self.t_models[agent].set_weights(q_model.get_weights())

    def save_models(self, suffix=""):
        for agent in self.q_models.keys():
            q_filename = "./models/" + agent + suffix + "_q-model.keras"
            self.q_models[agent].save(q_filename)
            t_filename = "./models/" + agent + suffix + "_t-model.keras"
            self.t_models[agent].save(t_filename)

    def load_models(self, suffix=""):
        # IMPORTANT: Assumes pre-trained models exist for every present agent
        for agent in self.q_models.keys():
            q_filename = "./models/" + agent + suffix + "_q-model.keras"
            t_filename = "./models/" + agent + suffix + "_t-model.keras"
            self.q_models[agent] = keras.saving.load_model(q_filename)
            self.t_models[agent] = keras.saving.load_model(t_filename)

