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


def time_now(micro=False):
    x = datetime.now()
    if micro:
        retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}_{x.microsecond}'
    else:
        retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}'

    return retval


def process_state(img):
    _img = np.mean(img, axis=2, dtype=np.uint8)
    _img = _img[::2, ::2]
    _img = np.where(_img == 0, 0, 255).astype(np.uint8)
    return _img


def ray_trace(seq_images):
    if len(seq_images.shape) < 3:  # single image
        return seq_images

    if len(seq_images) == 1:
        return seq_images[0]

    process_seq = []

    min_alpha = 0.25
    max_alpha = 1.0
    seq_alpha = np.linspace(min_alpha, max_alpha, num=len(seq_images))
    print(time_now(), seq_alpha)
    for alpha, image in zip(seq_images, seq_alpha):
        s = image * alpha
        process_seq.append(s)

    process_seq = np.array(process_seq, dtype=np.uint8)
    ray_trace_image = np.max(process_seq, axis=0)
    return ray_trace_image


class DQNMemory:
    def __init__(self, size=10_000):
        self.size = size
        # s and s' will be store sequentially
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminals = []  # when an episode ends, there is no next action, reward

    def pop(self):
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.terminals.pop(0)

    def get_state(self, idx, state_count=4, next_state=False):
        """
        Return the state at `idx` and the last `state_count` states.

        If the `idx` state is a terminal state and this state is not going
        to be used as `next_state` then return `None`.
        """
        # get the positive index (e.g. len = 4, idx = -1 => idx = 3)
        idx = idx % len(self.states)

        if not next_state:
            if idx >= len(self.actions):
                return None

            elif self.actions[idx] == None:
                return None

        if idx == 0:  # there is no state before `idx`
            return np.array([self.states[0]])

        state_stack = []
        for i in range(state_count):
            # step backward to retrieve the state
            _idx = idx - i

            # print('_idx:', _idx)
            if _idx <= 0:
                break
            # hit terminal state of the last episode
            if _idx >= len(self.actions):
                pass
            elif self.actions[_idx] == None:
                break

            state_stack.insert(0, self.states[_idx])

        if len(state_stack) == 0:
            return None

        retval = np.array(state_stack)
        return retval

    def start_eps(self, state):
        """
        When starting and episode (`env.reset`), 
        we will only receive the initial state.
        There is no reward or action yet.
        """
        self.states.append(state)

    def end_eps(self):
        """
        When an episode ends, the size of `self.actions`, `self.rewards`, 
        and `self.terminals` is lag behind because of the episode's first state 
        is append without reward, action, and terminal.
        """
        self.actions.append(None)
        self.rewards.append(None)
        self.terminals.append(None)

    def add(self, new_state, action, reward, terminal):
        """
        This method should be call after the agent has taken an action.
        """
        if len(self.states) > self.size:
            self.pop()

        self.states.append(new_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def sample_memory(self, size=1_000):
        idx_pool = list(range(len(self.states) - 1))

        shuffle_indices = []
        for _ in range(size):
            if len(idx_pool) == 0:
                break
            select_idx = random.choice(idx_pool)
            idx_pool.remove(select_idx)
            shuffle_indices.append(select_idx)

        state_list = [ray_trace(self.get_state(idx)) for idx in shuffle_indices]
        action_list = [self.actions[idx] for idx in shuffle_indices]
        reward_list = [self.rewards[idx] for idx in shuffle_indices]
        next_state_list = [ray_trace(self.get_state(idx)) for idx in shuffle_indices]

        return state_list, action_list, reward_list, next_state_list


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

    def replay_memory(self, state_list: list, action_list: list, reward_list: list, next_state_list: list):
        batch = state_list + reward_list
        batch = np.array(batch, dtype=np.uint8)
        batch = self.reshape_states(batch)
        batch = batch.astype(np.float) / 255.0
        q_result = self.model.predict(batch)
        current_q_values = q_result[:len(state_list)]
        next_state_q_values = q_result[len(state_list):]
