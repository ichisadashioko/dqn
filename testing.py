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
from tensorflow import keras

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
    replay_memory_size = 1_000  # SGD updates are sampled from this number of most recent frames
    agent_history_length = 4  # the number of most recent frames experienced by agent that are given as input to the Q network
    action_repeat = 4  # repeat each action selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4 input frame

    env_name = 'Breakout-v0'
    # general setup
    env = gym.make(env_name)
    n_actions = env.action_space.n

    model_filename = f'{env_name}/{env_name}.h5'

    assert os.path.exists(model_filename)

    print('Loading saved model...')
    model = tf.keras.models.load_model(model_filename)

    agent = DQNAgent(
        n_actions=n_actions,
        replay_memory=replay_memory_size,
        hist_len=agent_history_length,
        max_reward=1,
        min_reward=-1,
        network=model,
        action_repeat=action_repeat,
    )

    # testing loop
    num_steps = 1_000

    screen = env.reset()
    reward = 0
    terminal = 0

    for step in tqdm(range(num_steps)):
        env.render()
        time.sleep(1 / 60.0)

        action = agent.perceive(
            reward, screen, terminal,
            testing=True,
            # testing_ep=0.2,
            testing_ep=0,
        )

        # game over? get next game!
        if not terminal:
            observation = env.step(action)
            screen, reward, done, info = observation
            if done:
                terminal = 1
        else:
            screen = env.reset()
            reward = 0
            terminal = 0

    env.close()