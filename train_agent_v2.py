import os
import time
import math
import random

from tqdm import tqdm

import numpy as np
import gym
import matplotlib.pyplot as plt

from TransitionTableV2 import TransitionTable
from DQNAgentV2 import DQNAgent

env_name = 'Breakout-v0'
env = gym.make(env_name)
n_actions = env.action_space.n

num_steps = 20_000
frame_skip = 4

agent = DQNAgent(n_actions)

# Training loop
terminal = 1

for step in tqdm(range(num_steps)):
    if terminal:
        state = env.reset()

    # agent will take action every `frame_skip` frames
    if step % frame_skip == 0:
        action = agent.perceive(state)

    s2, reward, done, info = env.step(action)

    terminal = 1 if done else 0
