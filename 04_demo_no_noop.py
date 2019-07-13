import os
import time
import math
import random

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import gym

import tensorflow as tf
from tensorflow import keras

import dqn_utils
from dqn_utils import *


if __name__ == "__main__":
    env_name = 'Breakout-v0'
    env = gym.make(env_name)
    # n_actions = env.action_space.n
    n_actions = 3

    model_filename = f'{env_name}_NO_NOOP.h5'
    assert os.path.exists(model_filename)
    model = keras.models.load_model(model_filename)

    agent = DQN(model=model, n_actions=n_actions)

    dqn_memory = DQNMemory(size=500)

    num_episodes = 10
    try:
        for episode in range(num_episodes):

            done = False
            state = env.reset()
            state = process_state(state)
            dqn_memory.start_eps(state)
            env.render()

            total_reward = 0

            while not done:
                batch = dqn_memory.get_state(-1, next_state=True)
                batch = ray_trace(batch)
                batch = agent.reshape_state(batch)
                batch = batch / 255.0
                action = np.argmax(agent.model.predict(batch)[0])

                action += 1

                print('action:', action)
                observation = env.step(action)

                action -= 1

                env.render()

                new_state, reward, done, info = observation
                total_reward += reward
                # process new state to reduce memory
                new_state = process_state(new_state)
                # add new observation to `dqn_memory`
                dqn_memory.add(new_state, action, reward, done)

                state = new_state

            dqn_memory.end_eps()

            print(f'episode {episode} -- total reward: {total_reward}')
    except KeyboardInterrupt:
        pass
    env.close()
