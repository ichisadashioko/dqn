import os
import time
import math

import numpy as np
import matplotlib.pyplot as plt

import gym

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam


class DQNMemory:
    def __init__(self, size=10_000):
        self.size = size
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

    def pop(self):
        self.states.pop(0)
        self.next_states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)

    def add(self, state, action, reward, next_state):
        if len(self.states) > self.size:
            self.pop()

        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)

    def sample_memory(self):
        pass


class DQN:
    def __init__(self, input_shape=(105, 80, 1), n_actions=4, lr=0.00001, discount_rate=0.8):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.model = create_keras_model(input_shape, n_actions)
        self.lr = lr
        self.discount_rate = discount_rate

        optimizer = Adam(
            learning_rate=self.lr,
        )

        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['accuracy', ],
        )

    def create_keras_model(input_shape, n_actions):
        model = keras.Sequential([
            Conv2D(
                filters=32,
                kernel_size=8,
                strides=4,
                activation='relu',
                input_shape=(*input_shape, ),
                data_format='channels_last',
            ),
            Conv2D(
                filters=64,
                kernel_size=4,
                strides=2,
                activation='relu',
            ),
            Conv2D(
                filters=64,
                kernel_size=3,
                activation='relu',
            ),
            Flatten(),
            Dense(
                units=n_actions,
            ),
        ])

        return model

    def reshape_state(self, state):
        """
        `state`: 2D array (height, width)
        """
        return np.reshape(a=state, newshape=(1, *self.input_shape))

    def reshape_states(self, states):
        """
        `states`: 3D array (batch_size, height, width)
        """
        return np.reshape(a=states, newshape=(-1, *self.input_shape))

    def update_q_values(self, state, action, reward, new_state):
        # batch the current `state` and the `new_state` together for optimization
        q_batch = np.array([
            state,
            new_state,
        ])
        # reshape the batch
        q_batch = self.reshape_states(q_batch)
        # get the Q-values from the deep neural network
        q_results = self.model.predict(q_batch)
        current_q_values = q_results[0]
        new_state_q_values = q_results[1]

        # copy the current Q-values
        target_q_values = current_q_values
        # modify the Q-value of the `action`
        target_q_values[action] = reward + self.discount_rate * np.max(new_state_q_values)

        self.model.fit(
            x=self.reshape_state(state),
            y=target_q_values,
            verbose=0,
        )

    def replay_memory(self, size=100):
        pass
