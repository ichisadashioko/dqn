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


class DQNAgent:
    def __init__(
        self,
        n_actions=4,
        ep_start=1.0,
        ep_end=0.1,
        ep_endt=1_000_000,
        lr=0.00025,
        minibatch_size=1,
        valid_size=500,
        discount=0.99,
        update_freq=1,
        n_replay=1,
        learn_start=0,
        replay_memory=1_000_000,
        hist_len=1,
        max_reward=None,
        min_reward=-None,
    ):
        """
        Parameters
        ----------
        n_actions : int
            The number of actions that the agent can take.

        ep_start : float
            The inital epsilon value in epsilon-greedy.

        ep_end : float
            The final epsilon value in epsilon-greedy.

        ep_endt : int
            The number of timesteps over which the inital value of epislon is linearly annealed to its final value.

        lr : float
            The learning rate used by RMSProp.
        """
        # self.state_dim = state_dim
        self.n_actions = n_actions

        # epsilon annealing
        self.ep_start = ep  # inital epsilon value
        self.ep = self.ep_start  # exploration probability
        self.ep_end = ep_end  # final epsilon value
        self.ep_endt = ep_endt  # the number of timesteps over which the inital value of epislon is linearly annealed to its final value

        # learning rate anealing
        # self.lr_start = lr
        # self.lr = self.lr_start
        # self.lr_end = lr_end
        # self.lr_endt = lr_endt
        self.lr = lr
        # self.wc = wc # L2 weight cost
        self.minibatch_size = minibatch_size
        self.valid_size = valid_size

        # Q-learning paramters
        self.discount = discount  # discount factor
        self.update_freq = update_freq
        # number of points to replay per learning step
        self.n_replay = n_replay
        # number of steps after which learning starts
        self.learn_start = learn_start
        # size of the transition table
        self.replay_memory = replay_memory
        self.hist_len = hist_len
        self.max_reward = max_reward
        self.min_reward = min_reward

        # create transition table
        self.transitions = TransitionTable(histLen=self.hist_len, maxSize=self.replay_memory)

        self.numSteps = 0  # number of perceived states
        self.lastState = None
        self.lastAction = None
        # self.

    def reset(self, state):
        # TODO 9 Low-priority
        pass

    def preprocess(self, rawstate):  # DONE
        state = np.mean(rawstate, axis=2, dtype=np.uint8)
        state = state[::2, ::2]
        # turn grayscale image to binary image
        # _img = np.where(_img == 0, 0, 255).astype(np.uint8)
        return state

    def getQUpdate(self, s, a, r, s2, term):  # TODO 2
        # The order of calls to forward is a bit odd
        # in order to avoid unnecessary calls (we only need 2)

        # delta = r + (1 - terminal) * gamma * max_a Q(s2, a) - Q(s, a)
        term = (term * -1) + 1

    def qLearnMinibatch(self):  # TODO 4
        pass

    def sample_validation_data(self):  # TODO 4
        pass

    def sample_validation_statistics(self):  # TODO 4
        pass

    def perceive(self, reward, rawstate, terminal, testing=False, testing_ep=None):  # TODO 1
        # preprocess state
        state = self.preprocess(rawstate)
        # clip reward
        if self.max_reward is not None:
            reward = math.min(reward, self.max_reward)

        if self.min_reward is not None:
            reward = math.max(reward, self.min_reward)

        self.transitions.add_recent_state(state, terminal)

        currentFullState = self.transitions.get_recent()

        # store transition s, a, r, s'
        if self.lastState and not testing:
            self.transitions.add(self.lastState, self.lastAction, reward, self.lastTerminal)

    def eGreedy(self, state, testing_ep):  # TODO 3
        pass

    def greedy(self, state):  # TODO 6
        pass

    def createNetwork(self, input_shape=(105, 80, 4), n_actions=4):
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
                strides=1,
                activation='relu',
            ),
            Flatten(),
            Dense(
                units=512,
                activation='relu',
            ),
            Dense(
                units=n_actions,
            ),
        ])

        return model
