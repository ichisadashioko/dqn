import os
import time
import math
import random
from datetime import datetime

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import gym

import tensorflow as tf

from DQNAgent import DQNAgent
from TransitionTable import TransitionTable


def time_now(micro=False):
    x = datetime.now()
    if micro:
        retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}_{x.microsecond}'
    else:
        retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}'

    return retval


if __name__ == "__main__":

    # hyperparamters
    minibatch_size = 32  # number of training cases over which each stochastic gradient descent (SGD) update is computed
    replay_memory_size = 500_000  # SGD updates are sampled from this number of most recent frames
    agent_history_length = 4  # the number of most recent frames experienced by agent that are given as input to the Q network
    # target_network_update_frequency = 10_000  # the freuquency (measured in the number of parameter updates) with which the target netwrok is updated (this corresponds to the parameter C from Algorithm 1)
    discount_factor = 0.99  # discount factor gamma used in the Q-learning update
    # action_repeat = 4  # repeat each action selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4 input frame
    update_frequency = 4  # the number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates
    learning_rate = 0.00025  # the learning rate used by RMSProp
    # gradient_momentum = 0.95  # squared gradient (denominator) momentum used by RMSProp
    # min_squared_gradient = 0.01  # constant added to the squared gradient in the denominator of the RMSProp update
    inital_exploration = 1.0  # initial value of epsilon in epsilon-greedy exploration
    final_exploration = 0.1  # final value of epsilon in epsilon-greedy exploration
    final_exploration_frame = 1_000_000  # the number of frames over which the initial value of epsilon is linearly annealed to its final value
    replay_start_size = 50_000  # a uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory
    # no_op_max = 30  # maximum number of "do nothing" actions to be performed by agent at the start of an episode

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
        update_freq=update_frequency,
        learn_start=replay_start_size,
        replay_memory=replay_memory_size,
        hist_len=agent_history_length,
        max_reward=1,
        min_reward=-1,
    )

    num_steps = 100_000
    step = 0

    screen = env.reset()
    reward = 0
    terminal = 0

    train_start = time.time()

    for step in tqdm(range(num_steps)):
        action = agent.perceive(reward, screen, terminal)

        # game over? get next game!
        if not terminal:
            observation = env.step(action)
            # env.render()
            screen, reward, done, info = observation
            if done:
                terminal = 1
        else:
            screen = env.reset()
            reward = 0
            terminal = 0
