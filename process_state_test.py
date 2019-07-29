#! /usr/bin/env python3
from tqdm import tqdm

import numpy as np
import cv2
import gym


def process_state(state):
    s = state[::2, ::2, :]
    s = np.mean(s, axis=2, dtype=np.uint8)
    return s


# env_name = 'Breakout-v0'
env_name = 'Atlantis-v0'
env = gym.make(env_name)

num_steps = 1_000
terminal = 1

for step in tqdm(range(num_steps)):
    if terminal:
        state = env.reset()

    action = env.action_space.sample()

    _state = process_state(state)
    s2, reward, done, info = env.step(action)

    terminal = 1 if done else 0

    cv2.imshow('orig', state)
    cv2.imshow('processed', _state)

    zoom_state = cv2.resize(
        _state, 
        dsize=None, 
        fx=4, fy=4, 
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.imshow('zoom_state', zoom_state)

    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):
        break

    state = s2
cv2.destroyAllWindows()
