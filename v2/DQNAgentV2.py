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
from tensorflow.keras.optimizers import RMSprop

from TransitionTableV2 import TransitionTable


class DQNAgent:
    def __init__(
        self,
        n_actions=4,
        stateDim=(105, 80),
        histLen=4,
        memory_size=500_000,
        lr=0.0001,
        discount=0.99,
        ep_start=1.0,
        ep_end=0.1,
        ep_endt=1_000_000,
        learn_start=50_000,
        network=None,
    ):
        self.n_actions = n_actions
        self.stateDim = stateDim
        self.histLen = histLen
        self.memory_size = memory_size
        self.lr = lr
        self.discount = discount

        self.ep_start = ep_start
        self.ep = ep_start
        self.ep_end = ep_end
        self.ep_endt = ep_endt
        self.learn_start = learn_start

        self.numSteps = 0
        self.transitions = TransitionTable(
            histLen=self.histLen,
            maxSize=self.memory_size,
        )

        if network:
            self.network = network
        else:
            self.network = self.create_network(
                input_shape=(self.histLen, *self.stateDim),
                n_actions=self.n_actions,
                lr=self.lr,
            )

    def calc_epsilon(self):
        retval = (self.ep_start - self.ep_end)
        retval = retval * ((self.ep_endt - self.numSteps) / self.ep_endt)
        retval = max(retval, self.ep_end)
        return retval

    def perceive(self, state, testing=False):
        if not testing:
            return self.greedy(state)

        action = None
        self.ep = self.calc_epsilon()
        if random.random() < self.ep:
            action = random.randrange(self.n_actions)
        else:
            action = self.greedy(state)

        if self.numSteps > self.learn_start:
            # TODO update Q-values
            pass

        self.numSteps += 1
        return action

    def greedy(self, state):
        pass

    def create_network(self, input_shape, n_actions, lr):
        model = keras.Sequential([
            Conv2D(
                filters=32,
                kernel_size=8,
                strides=4,
                activation='relu',
                input_shape=(*input_shape, ),
                data_format='channels_first',
            ),
            Conv2D(
                filters=64,
                kernel_size=4,
                strides=2,
                activation='relu',
                data_format='channels_first',
            ),
            Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                activation='relu',
                data_format='channels_first',
            ),
            Flatten(),
            Dense(
                units=256,
                activation='relu',
            ),
            Dense(
                units=n_actions,
                activation='linear',
            ),
        ])

        optimizer = RMSprop(lr=lr)

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['accuracy', 'mse'],
        )

        model.summary()

        return model

    def process_state(self, state):
        s = state[::2, ::2, :]
        s = np.mean(s, axis=2)
        return s.astype(np.uint8)
