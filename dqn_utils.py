import os
import time
import math
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import gym

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam


def time_now():
    x = datetime.now()
    retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}'
    return retval


def process_image(img):
    _img = np.mean(img, axis=2, dtype=np.uint8)
    _img = _img[::2, ::2]
    return _img


def ray_trace(seq_images):

    if len(seq_images.shape) < 3:  # single image
        return seq_images
    process_seq = []

    for idx, image in enumerate(seq_images):
        # fade past images
        alpha = idx / len(seq_images)
        s = image * alpha
        process_seq.append(s)

    process_seq = np.array(process_seq, dtype=np.uint8)
    ray_trace_image = np.max(process_seq, axis=0)
    return ray_trace_image


class DQNMemory:
    def __init__(self, size=10_000):
        self.size = size
        self.states = []
        self.actions = []
        self.rewards = []

    def pop(self):
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)

    def add(self, state, action, reward):
        if len(self.states) > self.size:
            self.pop()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def sample_memory(self, size=1_000):

        pass

    def get_state(self, idx, frame_count=4):
        """Return the frame at `idx` and the last `frame_count` frames"""

        if (idx + len(self.states)) == 0:
            return np.array([self.states[0]])

        end = idx
        start = end - frame_count
        retval = np.array(self.states[start:end])
        return retval


class DQN:
    def __init__(self, model=None, input_shape=(105, 80, 1), n_actions=4, lr=0.00001, discount_rate=0.8):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.lr = lr
        self.discount_rate = discount_rate

        if model is None:
            self.model = self.create_keras_model(input_shape, n_actions)
            optimizer = Adam(
                learning_rate=self.lr,
            )

            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['accuracy', ],
            )
        else:
            self.model = model

    def create_keras_model(self, input_shape, n_actions):
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
        return np.reshape(a=states, newshape=(len(states), *self.input_shape))

    def update_q_values(self, state, action, reward, new_state):
        # batch the current `state` and the `new_state` together for optimization
        q_batch = np.array([
            state,
            new_state,
        ])
        # reshape the batch
        q_batch = np.reshape(a=q_batch, newshape=(2, *self.input_shape))
        # normalize values
        q_batch = q_batch / 255.0
        # get the Q-values from the deep neural network
        q_results = self.model.predict(q_batch)
        current_q_values = q_results[0]
        new_state_q_values = q_results[1]

        # copy the current Q-values
        target_q_values = current_q_values
        # modify the Q-value of the `action`
        target_q_values[action] = reward + self.discount_rate * np.max(new_state_q_values)
        target_q_values = np.array([target_q_values])
        x = self.reshape_state(state)
        x = x / 255.0
        self.model.fit(
            x=x,
            y=target_q_values,
            verbose=0,
        )

    def replay_memory(self, size=100):
        pass
