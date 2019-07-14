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

from TransitionTable import TransitionTable


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
        network=None,
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

        self.lr = lr
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

        self.network = network if network else self.createNetwork(n_actions=n_actions)

        # create transition table
        self.transitions = TransitionTable(histLen=self.hist_len, maxSize=self.replay_memory)

        self.numSteps = 0  # number of perceived states
        self.lastState = None
        self.lastAction = None
        self.lastTerminal = None

    def reset(self, state):
        # TODO 9 Low-priority
        pass

    def preprocess(self, rawstate):  # DONE
        state = np.mean(rawstate, axis=2, dtype=np.uint8)
        state = state[::2, ::2]
        # turn grayscale image to binary image
        # _img = np.where(_img == 0, 0, 255).astype(np.uint8)
        return state

    def getQUpdate(self, s, a, r, s2, term):  # DOME 2
        # The order of calls to forward is a bit odd
        # in order to avoid unnecessary calls (we only need 2)

        # delta = r + (1 - terminal) * gamma * max_a Q(s2, a) - Q(s, a)
        term = (term * -1) + 1

        target_q_net = self.network

        # compute max_a Q(s_2, a)
        q2_max = np.max(target_q_net.predict(s2), axis=1)

        # compute q2 = (1-terminal) * gamma * max_a Q(s2,a)
        q2 = q2_max * self.discount
        q2 = q2 * term

        delta = r + q2

        q_all = self.network.predict(s)
        q = np.zeros(len(q_all), dtype=np.float32)
        for i in range(len(q_all)):
            q[i] = q_all[i][a[i]]

        delta = delta + (q * -1)

        targets = np.zeros(shape=(self.minibatch_size, self.n_actions), dtype=np.float32)
        for i in range(min(self.minibatch_size, len(a))):
            targets[i][a[i]] = delta[i]

        return targets, delta, q2_max

    def qLearnMinibatch(self):  # DONE 4
        # perform a minibatch Q-learning update:
        # w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw

        # w = w + (gamma max Q(s2, a2) - Q(s,a)) # this is the label for Keras
        assert self.transitions.size() > self.minibatch_size

        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)

        targets, delta, q2_max = self.getQUpdate(s, a, r, s2, term)

        # DONE 2 what is `targets, delta, q2_max`
        # delta = r + (1-term) * gamma * max_a Q(s2, a) - Q(s, a)

        # targets.shape = (batch_size, n_action)
        # delta.shape = (batch_size)
        # q2_max.shape = (batch_size)

        self.network.fit(
            x=s,
            y=targets,
            epochs=1,
            batch_size=self.minibatch_size,
        )

    def sample_validation_data(self):  # TODO 8
        # for validation
        pass

    def sample_validation_statistics(self):  # TODO 8
        # for validation
        pass

    def perceive(self, reward, rawstate, terminal, testing=False, testing_ep=None):  # DONE 1
        """
        reward : number
            The received reward from environment.

        rawstate : ndarray
            The game screen.

        terminal : int
            If the game end then `terminal = 1` else `terminal = 0`.

        testing_ep : number
            Testing epsilon value for the epsilon-greedy algorithm.
        """
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

        curState = self.transitions.get_recent()  # DONE (4, 105, 80)
        curState = np.array([curState], dtype=np.uint8)

        # select action
        action = 0
        if not terminal:
            action = self.eGreedy(curState, testing_ep)

        # do some Q-learning updates
        if (self.numSteps > self.learn_start) and (not testing) and (self.numSteps % self.update_freq == 0):
            for i in range(self.n_replay):
                self.qLearnMinibatch()

        if not testing:
            self.numSteps += 1

        self.lastState = state
        self.lastAction = action
        self.lastTerminal = terminal

        return action

    def eGreedy(self, state, testing_ep=None):  # DONE 3
        if testing_ep is None:
            self.ep = self.ep_end + max(0, (self.ep_start - self.ep_end) * (self.ep_endt - max(0, self.numSteps - self.learn_start)) / self.ep_endt)
        else:
            self.ep = testing_ep

        if random.random() < self.ep:
            return random.randrange(0, self.n_actions)
        else:
            return self.greedy(state)

    def greedy(self, state):  # DONE 6
        q = self.network.predict(state)[0]
        max_q = q[0]
        best_a = [0]

        # evaluate all other actions (with random tie-breaking)
        for a in range(1, self.n_actions):
            if q[a] > max_q:
                best_a = [a]
                max_q = q[a]
            elif q[a] == max_q:
                best_a.append(a)

        r = random.randrange(0, len(best_a))

        self.lastAction = best_a[r]

        return best_a[r]

    def createNetwork(self, input_shape=(4, 105, 80), n_actions=4):
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
