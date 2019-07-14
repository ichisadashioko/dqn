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


def time_now(micro=False):
    x = datetime.now()
    if micro:
        retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}_{x.microsecond}'
    else:
        retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}'

    return retval


def create_model(input_shape=(105, 80, 4), n_actions=4):
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


def compile_model(model, lr=0.00025):
    optimizer = RMSprop(lr=lr)
    model.compile(
        optimizer=optimizer,
    )


class TransitionTable:
    def __init__(
        self,
        stateDim=(105, 80),
        histLen=1,
        maxSize=1_000_000,
        bufferSize=1024,
    ):
        self.stateDim = stateDim
        self.histLen = histLen
        self.maxSize = maxSize
        self.bufferSize = bufferSize
        self.buf_ind = None

        self.recentMemSize = self.histLen

        self.numEntries = 0
        self.insertIndex = 0

        # DONE pre-allocate (maxSize, dims) Tensors
        self.s = np.zeros(shape=(self.maxSize, *self.stateDim), dtype=np.uint8)
        self.a = np.zeros(self.maxSize, dtype=np.uint8)
        self.r = np.zeros(self.maxSize, dtype=np.float32)
        self.t = np.zeros(self.maxSize, dtype=np.uint8)

        # Tables for storing the last `histLen` states. They are used for constructing the most recent agent state more easily
        self.recent_s = []
        self.recent_a = []
        self.recent_t = []

        # DONE pre-allocate Tensors
        s_size = (self.histLen, *self.stateDim) # TODO 3 consider between 'channels_first' or 'channels_last'
        self.buf_a = np.zeros(self.bufferSize, dtype=np.uint8)
        self.buf_r = np.zeros(self.bufferSize, dtype=np.float32)
        self.buf_term = np.zeros(self.bufferSize, dtype=np.uint8)
        # TODO 4 check the buffer shape before pass it to the model
        self.buf_s = np.zeros(s_size, dtype=np.uint8)
        self.buf_s2 = np.zeros(s_size, dtype=np.uint8)

    def reset(self):  # DONE
        self.numEntries = 0
        self.insertIndex = 0

    def size(self):  # DONE
        return self.numEntries

    def empty(self):  # DONE
        return self.numEntries == 0

    def fill_buffer(self):  # TODO 3
        assert self.numEntries >= self.bufferSize
        # clear CPU buffers
        self.buf_ind = 1

        for buf_ind in range(self.bufferSize):
            s, a, r, s2, term = self.sample_one()
            self.buf_s[buf_ind] = s
            self.buf_a[buf_ind] = a
            self.buf_r[buf_ind] = r
            self.buf_s2[buf_ind] = s2
            self.buf_term[buf_ind] = term

        self.buf_s = self.buf_s / 255.0
        self.buf_s2 = self.buf_s2 / 255.0

    def sample_one(self):  # TODO 3
        assert self.numEntries > 1

        valid = False
        while not valid:
            # start at 2 because of previous action
            index = random.randrange(2, self.numEntries - self.recentMemSize)

            if self.t[index + self.recentMemSize - 1] == 0:
                valid = True

        return self.get(index)

    def sample(self, batch_size=1):  # TODO 4
        assert batch_size < self.bufferSize

    def concatFrames(self, index, use_recent=False):  # TODO 4
        if use_recent:
            s, t = self.recent_s, self.recent_t
        else:
            s, t = self.s, self.t

        fullstate = None

        # Copy frames from the current episode
        # TODO 5 copy frames and zero-out un-related frames
        # for i in range(index, )
        return fullstate

    def concatActions(self, index, use_recent=False):  # TODO 4
        act_hist = []
        if use_recent:
            a, t = self.recent_a, self.recent_t
        else:
            a, t = self.a, self.t

        # zero out frames from all but the most recent episode
        # TODO 5

    def get_recent(self):  # DONE
        # Assumes that the most recent state has been added, but the action has not
        return self.concatFrames(1, True)

    def get(self, index):  # DONE
        s = self.concatFrames(index)
        s2 = self.concatFrames(index + 1)
        ar_index = index + self.recentMemSize - 1

        return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index + 1]

    def add(self, s, a, r, term):  # DONE
        # Increment until at full capacity
        if self.numEntries < self.maxSize:
            self.numEntries += 1

        # Always insert at next index, then wrap around
        self.insertIndex += 1
        # Overwrite oldest experience once at capacity
        if self.insertIndex > self.maxSize:
            self.insertIndex = 1

        # Overwrite (s, a, r, t) at `insertIndex`
        self.s[self.insertIndex] = s
        self.a[self.insertIndex] = a
        self.r[self.insertIndex] = r
        if term:
            self.t[self.insertIndex] = 1
        else:
            self.t[self.insertIndex] = 0

    def add_recent_state(self, s, term):  # TODO
        if len(self.recent_s) == 0:
            for i in range(self.recentMemSize):
                pass

    def add_recent_action(self, a):  # TODO
        pass


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
        # self.clip_delta = clip_delta
        # self.target_q = target_q
        # self.best_q = 0

        self.transitions = TransitionTable(histLen=self.hist_len, maxSize=self.replay_memory)

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

    def perceive(self, reward, rawstate, terminal, testing, testing_ep):  # TODO 1
        state = self.preprocess(rawstate)

        if self.max_reward is not None:
            reward = math.min(reward, self.max_reward)

        if self.min_reward is not None:
            reward = math.max(reward, self.min_reward)

        self.transitions.add_recent_state(state, terminal)

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


if __name__ == "__main__":
    # input_shape = (84, 84, 4)
    # model = create_model(input_shape=input_shape)
    # model = create_model()
    # model.summary()

    # hyperparamters
    minibatch_size = 32  # number of training cases over which each stochastic gradient descent (SGD) update is computed
    replay_memory_size = 1_000_000  # SGD updates are sampled from this number of most recent frames
    agent_history_length = 4  # the number of most recent frames experienced by agent that are given as input to the Q network
    target_network_update_frequency = 10_000  # the freuquency (measured in the number of parameter updates) with which the target netwrok is updated (this corresponds to the parameter C from Algorithm 1)
    discount_factor = 0.99  # discount factor gamma used in the Q-learning update
    action_repeat = 4  # repeat each action selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4 input frame
    update_frequency = 4  # the number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates
    learning_rate = 0.00025  # the learning rate used by RMSProp
    gradient_momentum = 0.95  # squared gradient (denominator) momentum used by RMSProp
    min_squared_gradient = 0.01  # constant added to the squared gradient in the denominator of the RMSProp update
    inital_exploration = 1.0  # initial value of epsilon in epsilon-greedy exploration
    final_exploration = 0.1  # final value of epsilon in epsilon-greedy exploration
    final_exploration_frame = 1_000_000  # the number of frames over which the initial value of epsilon is linearly annealed to its final value
    replay_start_size = 50_000  # a uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory
    no_op_max = 30  # maximum number of "do nothing" actions to be performed by agent at the start of an episode

    env_name = 'Breakout-v0'
    # general setup
    env = gym.make(env_name)
    n_actions = env.action_space.n

    agent = DQNAgent(
        n_actions=n_actions,
        ep_start=inital_exploration,
        ep_end=final_exploration,
        ep_endt=final_exploration_frame,
        lr=learning_rate,
        minibatch_size=minibatch_size,
        discount=discount_factor,
        update_freq=target_network_update_frequency,
        learn_start=replay_start_size,
        replay_memory=replay_memory_size,
        hist_len=agent_history_length,
        max_reward=1,
        min_reward=-1,
    )
