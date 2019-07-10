import os
import time
import math
import random
import threading

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
    n_actions = env.action_space.n

    model_filename = f'{env_name}.h5'
    if os.path.exists(model_filename):
        model = keras.models.load_model(model_filename)

        agent = DQN(model=model, n_actions=n_actions)
    else:
        agent = DQN(n_actions=n_actions)

    dqn_memory = DQNMemory()

    num_episodes = 10
    max_epsilon = 1.0  # 100% random
    min_epsilon = 0.4
    delta_epsilon = (max_epsilon - min_epsilon) / num_episodes

    try:
        for episode in tqdm(range(num_episodes)):
            # for episode in range(num_episodes):
            eps_epsilon = max_epsilon - delta_epsilon * episode
            eps_epsilon = max(eps_epsilon, min_epsilon)

            done = False
            state = env.reset()
            state = process_state(state)
            dqn_memory.states.append(state)
            env.render()

            while not done:
                if random.uniform(0, 1) < eps_epsilon:
                    action = env.action_space.sample()

                    greedy = False
                else:
                    batch = dqn_memory.get_state(-1)
                    batch = ray_trace(batch)
                    batch = agent.reshape_state(batch)
                    batch = batch / 255.0
                    action = np.argmax(agent.model.predict(batch)[0])

                    greedy = True

                observation = env.step(action)
                env.render()

                new_state, reward, done, info = observation
                # process new state to reduce memory
                new_state = process_state(new_state)
                # add new observation to `dqn_memory`
                dqn_memory.add(new_state, action, reward)

                _state = ray_trace(dqn_memory.get_state(-2))
                _new_state = ray_trace(dqn_memory.get_state(-1))

                agent.update_q_values(_state, action, reward, _new_state)

            #     break
            # break
            if not os.path.exists(env_name):
                os.makedirs(env_name)
            weights_filename = f'{env_name}/{env_name}_weights_{time_now()}_eps_{episode}.h5'
            agent.model.save_weights(weights_filename)

    except KeyboardInterrupt:
        pass

    env.close()
    agent.model.save(model_filename)