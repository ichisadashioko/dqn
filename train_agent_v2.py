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
state = env.reset()
terminal = 0

# agent will take action every 4 frames
for step in tqdm(range(num_steps)):
    if step % frame_skip == 0:
        action = agent.perceive(state)