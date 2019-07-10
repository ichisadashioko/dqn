import os
import time

import numpy as np
import cv2

import gym

from dqn_utils import *


def cv_img(img):
    img = cv2.resize(img, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    # return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


if __name__ == "__main__":
    env_name = 'Breakout-v0'
    env = gym.make(env_name)

    dqn_memory = DQNMemory()

    num_episodes = 10

    is_break = False
    try:
        for eps in range(num_episodes):
            done = False
            state = env.reset()
            state = process_state(state)
            dqn_memory.start_eps(state)
            while not done:
                # for _ in range(1_000):
                action = env.action_space.sample()

                observation = env.step(action)
                new_state, reward, done, info = observation
                new_state = process_state(new_state)

                dqn_memory.add(new_state, action, reward, done)
                # env.render()

                # retrieve_state = dqn_memory.get_state(-1)
                # if retrieve_state is None:
                #     continue

                cv2.imshow(env_name, cv_img(ray_trace(dqn_memory.get_state(-1, next_state=True))))
                k = cv2.waitKey(0) & 0xff
                if k == ord('q'):
                    is_break = True
                    break
                    
                if k == ord('n'):
                    break

            dqn_memory.end_eps()
            if is_break:
                break


    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
