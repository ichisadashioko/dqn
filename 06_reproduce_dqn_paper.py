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


def process_state(img):
    _img = np.mean(img, axis=2, dtype=np.uint8)
    _img = _img[::2, ::2]
    # _img = np.where(_img == 0, 0, 255).astype(np.uint8)
    return _img


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
    def __init__(self, histLen=1, maxSize=1_000_000):
        self.histLen = histLen
        self.maxSize = maxSize

        self.recentMemSize = self.histLen

        self.numEntries = 0
        self.insertIndex = 0

        # TODO pre-allocate (maxSize, dims) Tensors
        self.s = []
        self.a = []
        self.r = []
        self.t = []
        # Tables for storing the last `histLen` states. They are used for constructing the most recent agent state more easily

        self.recent_s = []
        self.recent_a = []
        self.recent_t = []

    def reset(self):
        self.numEntries = 0
        self.insertIndex = 0
    
    def size(self):
        return self.numEntries
    
    def empty(self):
        return self.numEntries == 0
    
    def add(self, s, a, r, term):
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
    
    def concatFrames(self, index, use_recent=False):
        if use_recent:
            s, t = self.recent_s, self.recent_t
        else:
            s, t = self.s, self.t
        
        fullstate = None

        # Copy frames from the current episode
        # TODO copy frames and zero-out un-related frames
        # for i in range(index, )
        return fullstate

    def get(self, index):
        s = self.concatFrames(index)
        s2 = self.concatFrames(index+1)
        ar_index = index + self.recentMemSize-1

        return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1]
        
    def sample_one(self):
        assert self.numEntries > 1

        valid = False
        while not valid:
            # start at 2 because of previous action
            index = random.randrange(2, self.numEntries-self.recentMemSize)

            if self.t[index+self.recentMemSize-1] == 0: # 
                valid = True
        
        return self.get(index)

    def fill_buffer(self):
        # TODO fill_buffer
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
        max_reward=1,
        min_reward=-1,
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
        self.ep_start = ep # inital epsilon value
        self.ep = self.ep_start # exploration probability
        self.ep_end = ep_end # final epsilon value
        self.ep_endt = ep_endt # the number of timesteps over which the inital value of epislon is linearly annealed to its final value

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
        self.discount = discount # discount factor
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

    agent = DQNAgent()
