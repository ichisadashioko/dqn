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
        valid_size=512,
        discount=0.99,
        update_freq=1,
        n_replay=1,
        learn_start=0,
        replay_memory=1_000_000,
        hist_len=1,
        max_reward=None,
        min_reward=None,
        network=None,
        action_repeat=4,
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
        self.ep_start = ep_start  # inital epsilon value
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

        if network:
            self.target_network = network
        else:
            self.target_network = self.createNetwork(
                n_actions=self.n_actions,
                lr=self.lr,
            )
        
        self.network = self.createNetwork(
            n_actions=self.n_actions,
            lr=self.lr,
        )

        # self.network.load_weights(self.target_network.get_weights())
        self.copy_weights(self.target_network, self.network)

        # create transition table
        self.transitions = TransitionTable(histLen=self.hist_len, maxSize=self.replay_memory)

        self.numSteps = 0  # number of perceived states
        self.lastState = None
        self.lastAction = None
        self.lastTerminal = None

        self.valid_s = None
        self.valid_a = None
        self.valid_r = None
        self.valid_s2 = None
        self.valid_term = None

        self.action_repeat = action_repeat

    def createNetwork(self, input_shape=(4, 105, 80), n_actions=4, lr=0.00025):
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
                units=512,
                activation='relu',
            ),
            Dense(
                units=n_actions,
                activation='linear',
            ),
        ])

        optimizer = RMSprop(lr=lr)

        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy', 'mse'],
        )

        return model

    def copy_weights(self, a, b):
        temp_weights_filename = 'temp_weights.h5'
        a.save_weights(temp_weights_filename)
        b.load_weights(temp_weights_filename)

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
        # merge `s` and `s2` together for one forward pass

        term = (term * -1) + 1
        # `s` and `s2` have to have the same shape
        assert s.shape == s2.shape
        forward_batch = np.concatenate((s, s2), axis=0)

        # I only scale values between [0..1] at the last step to reduce memory usage
        forward_batch = forward_batch / 255.0
        # We use the target_network to predict the Q-values
        q_batch = self.target_network.predict(forward_batch)
        mid_point = s.shape[0]

        s_q_values = q_batch[:mid_point]
        s2_q_values = q_batch[mid_point:]

        # compute max_a Q(s_2, a)
        s2_q_max = np.max(s2_q_values, axis=1)

        target_q_values = s2_q_max * self.discount
        target_q_values = target_q_values + r

        delta = []
        for i in range(len(a)):
            # calculate losses (for validation purpose only)
            delta.append(target_q_values[i] - s_q_values[i][a[i]])
            # update target q values
            # set all terminal state-action reward to 0
            s_q_values[i][a[i]] = target_q_values[i] * ((term[i] - 1) + 1)

        delta = np.array(delta)
        return s_q_values, delta, s2_q_max

    def qLearnMinibatch(self, verbose=0):
        # TODO accumulate losses instead of update rightaway
        # Perform a minibatch Q-learning update:
        assert self.transitions.size() > self.minibatch_size

        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        target_q_values, delta, s2_q_max = self.getQUpdate(s, a, r, s2, term)

        # We update the unstable network.
        self.network.fit(
            x=(s / 255.0),
            y=target_q_values,
            epochs=1,
            batch_size=self.minibatch_size,
            verbose=verbose,
        )

    def sample_validation_data(self):  # DONE 9
        # sample data for validation
        s, a, r, s2, term = self.transitions.sample(self.valid_size)
        self.valid_s = s
        self.valid_a = a
        self.valid_r = r
        self.valid_s2 = s2
        self.valid_term = term

    def compute_validation_statistics(self):  # DONE 9
        # We only sample validation data once to reduce computation.
        if self.valid_s is None:
            self.sample_validation_data()

        # for validation
        target_q_values, delta, s2_q_max = self.getQUpdate(
            s=self.valid_s,
            a=self.valid_a,
            r=self.valid_r,
            s2=self.valid_s2,
            term=self.valid_term,
        )
        avg_loss = delta.mean()

        return avg_loss

    def perceive(self, reward, rawstate, terminal, testing=False, testing_ep=None, verbose=0):  # DONE 1
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
            reward = min(reward, self.max_reward)

        if self.min_reward is not None:
            reward = max(reward, self.min_reward)

        self.transitions.add_recent_state(state, terminal)

        # store transition s, a, r, s'
        if (self.lastState is not None) and not testing:
            self.transitions.add(self.lastState,self.lastAction,reward,self.lastTerminal)

        # select action
        action = 0
        if not terminal and self.numSteps % self.action_repeat == 0:
            curState = self.transitions.get_recent()  # curState.shape == (4, 105, 80)
            # convert to batch (1, 4, 105, 80)
            curState = np.array([curState], dtype=np.uint8)

            action = self.eGreedy(curState, testing_ep)
        else:
            action = self.lastAction
        
        if not testing:
            if (self.numSteps > self.learn_start) and (self.numSteps % self.update_freq == 0):
                # do some Q-learning updates
                for _ in range(self.n_replay):
                    self.qLearnMinibatch(verbose=verbose)

        self.numSteps += 1

        self.lastState = state
        self.lastAction = action
        self.lastTerminal = terminal

        return action

    def eGreedy(self, state, testing_ep=None):  # DONE 3
        """
        testing_ep : testing epsilon
        """
        if testing_ep is None:
            ep_range = self.ep_start - self.ep_end
            ep_prog = 1.0 - max(0, self.numSteps - self.learn_start) / self.ep_endt
            ep_delta = ep_range * ep_prog
            self.ep = self.ep_end + max(0, ep_delta)
        else:
            self.ep = testing_ep

        if random.random() < self.ep:
            return random.randrange(0, self.n_actions)
        else:
            return self.greedy(state, testing_ep)

    def greedy(self, state, testing=None):  # DONE 6
        q = self.network.predict(state / 255.0)[0]
        max_q = q[0]
        best_a = [0]

        # evaluate all other actions (with random tie-breaking)
        for a in range(1, self.n_actions):
            if q[a] > max_q:
                best_a = [a]
                max_q = q[a]
            elif q[a] == max_q:
                best_a.append(a)
        # random tie-breaking
        r = random.randrange(0, len(best_a))
        self.lastAction = best_a[r]

        if testing is not None:
            print(f'numSteps: {self.numSteps}')
            print(f'action: {self.lastAction}')
            print(q)

        return best_a[r]