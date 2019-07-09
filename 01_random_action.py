import os
import time

import numpy as np
import cv2

import gym

def cv_img(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    env_name = 'Breakout-v0'
    env = gym.make(env_name)
    done = False
    state = env.reset()

    # cv2.imshow(env_name, cv_img(state))

    # while not done:
    for _ in range(1_000):
        # action = env.action_space.sample()
        # actions [NOOP, FIRE, RIGHT, LEFT]
        # references 'atari_py/ale_interface/src/games/supported/Breakout.cpp:83'
        # player needs to fire to fire the ball
        action = 1
        observation = env.step(action)
        state, reward, done, info = observation

        env.render()

    #     cv2.imshow(env_name, cv_img(state))
    #     cv2.waitKey(20)

    # cv2.destroyAllWindows()