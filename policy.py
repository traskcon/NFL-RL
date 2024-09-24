import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import copy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Policy():
    # NOTE: Each agent should have its own policy, observations in order to minimize Q_table size 
    # for full simulation. Initial example uses a common q_table for all agents
    def __init__(self, env, observations):
        # Environment state is a tuple of n 1x2 np arrays containing each agent's position
        self.env = env
        self.reward_function = env.scenario.reward
        self.models = dict()
        for agent, state in observations.items():
            self.model[agent] = self.build_dqn(state)

    def choose_action(self, agent):
        # Environment state is a tuple of n 1x2 np arrays containing each agent's position
        # Initial basic decision-making policy:
        # Choose action that gives the largest reward based on the current state
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
        model.add(keras.layers.Dense(24, input_shape=state.shape, activation="relu",kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation="relu", kernel_initializer=init))
        model.add(keras.layers.Dense(action_shape, activation="linear", kernel_initializer=init))
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])
        return model


