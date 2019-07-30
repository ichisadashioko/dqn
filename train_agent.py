import os
import time
import math
import random
from datetime import datetime

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.envs.registration import register

# disable TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    minibatch_size = 32
    replay_memory_size = 200_000
    agent_history_length = 4
    target_network_update_frequency = 10_000
    val_freq = target_network_update_frequency
    discount_factor = 0.99
    action_repeat = 4  # repeat each action selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4 input frame
    update_frequency = 4  # the number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates
    learning_rate = 0.001
    inital_exploration = 1.0  # initial value of epsilon in epsilon-greedy exploration
    final_exploration = 0.1  # final value of epsilon in epsilon-greedy exploration
    final_exploration_frame = 1_000_000  # the number of frames over which the initial value of epsilon is linearly annealed to its final value
    replay_start_size = 5_000  # a uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory

    # env_name = 'Breakout-v0'
    env_name = 'BreakoutNoFrameskip-v4'
    
    # configure model directory
    save_dir = env_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load model to continue training (optional)
    model = None
    model_filename = f'{env_name}/{env_name}.h5'
    if os.path.exists(model_filename):
        print('Loading saved model...')
        model = tf.keras.models.load_model(model_filename)

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
        network=model,
        action_repeat=action_repeat,
    )

    # training loop
    ep_reward_log = []
    # 100.0 fps on GTX 1050 (mobile) - utilize ~24%
    num_steps = 360_000
    step = 0

    screen = env.reset()
    reward = 0
    terminal = 0

    total_ep_reward = 0

    train_start = time.time()

    try:
        for step in tqdm(range(num_steps)):
            total_ep_reward += reward
            action = agent.perceive(reward, screen, terminal)

            # game over? get next game!
            if not terminal:
                observation = env.step(action)
                # env.render()
                screen, reward, done, info = observation
                if done:
                    terminal = 1
            else:
                ep_reward_log.append(total_ep_reward)
                # print('Last episode reward:', total_ep_reward)
                screen = env.reset()
                reward = 0
                terminal = 0

                total_ep_reward = 0

            if step % target_network_update_frequency == 0:
                # update the target network weights
                # agent.target_network.load_weights(agent.network.get_weights())
                agent.copy_weights(agent.network, agent.target_network)

            if step % val_freq == 0 and agent.transitions.numEntries > agent.transitions.bufferSize:
                avg_loss = agent.compute_validation_statistics()
                print('avg_loss:', avg_loss)
                w_filepath = f'{save_dir}/{env_name}_weights_{time_now()}_loss_{avg_loss:.2f}.h5'
                agent.target_network.save_weights(w_filepath)

    except KeyboardInterrupt:
        pass

    avg_loss = agent.compute_validation_statistics()
    model_filepath = f'{save_dir}/{env_name}_model_{time_now()}_loss_{avg_loss:.2f}.h5'
    agent.target_network.save(model_filepath)

    # save model in order to resume training
    agent.target_network.save(model_filename)

    np_reward_log = np.array(ep_reward_log)
    plt.plot(np_reward_log)
    plt.show()