import os
import time
import math
import random
from datetime import datetime
from collections import namedtuple
import argparse

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2

import gym
from gym.envs.registration import register

# I parse arguments before import TensorFlow because it takes forever to load TensorFlow.
parser = argparse.ArgumentParser(description="Testing script for Cartpole environment.")
parser.add_argument(
    '-inp', '--inp',
    dest='inp',
    type=str,
    required=True,
    help='input model filepath',
)
parser.add_argument(
    '-num', '--num',
    dest='num',
    type=int,
    required=False,
    help='the number of test episodes',
    default=4,
)

args = parser.parse_args()

model_filepath = args.inp
assert os.path.exists(model_filepath)

# disable GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# disable TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop

env_name = 'CartPole-v0'
# env = gym.make(env_name)
env = gym.make(env_name).unwrapped

target_net = keras.models.load_model(model_filepath)
target_net.summary()

num_episodes = args.num
max_steps = 2_000

reward_log = []
action_log = []

for episode in tqdm(range(num_episodes)):
    # env.seed(0)
    state = env.reset()

    total_reward = 0
    ep_action_log = []

    for _ in range(max_steps):
        action = np.argmax(target_net.predict(np.array([state]))[0])
        ep_action_log.append(action)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        screen = env.render(mode='rgb_array')
        screen = cv2.putText(
            img=screen,
            text=f'Episode {episode}',
            org=(5, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        screen = cv2.putText(
            img=screen,
            text=f'Total reward: {total_reward}',
            org=(5, 65),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        cv2.imshow('frame', screen)
        cv2.waitKey(10)
        state = next_state

        if done:
            break

    reward_log.append(total_reward)
    action_log.append(ep_action_log)


cv2.destroyAllWindows()
env.close()

reward_log = np.array(reward_log)
reward_mean = np.mean(reward_log)
reward_max = np.max(reward_log)
reward_min = np.min(reward_log)

print()
print(f'Mean: {reward_mean}')
print(f'Max: {reward_max}')
print(f'Min: {reward_min}')

mean_plot = np.zeros_like(reward_log)
mean_plot.fill(reward_mean)

plt.plot(reward_log)
plt.plot(mean_plot)
plt.show()
