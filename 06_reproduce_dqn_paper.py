import os
import time
import math
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import gym

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop


def time_now(micro=False):
    x = datetime.now()
    if micro:
        retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}_{x.microsecond}'
    else:
        retval = f'{x.year:04d}{x.month:02d}{x.day:02d}_{x.hour:02d}{x.minute:02d}{x.second:02d}'

    return retval


def process_state(img):
    _img = np.mean(img, axis=2, dtype=np.uint8)
    _img = _img[::2, ::2]
    # _img = np.where(_img == 0, 0, 255).astype(np.uint8)
    return _img


def create_model(input_shape=(105, 80, 4), n_actions=4):
    model = keras.Sequential([
        Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation='relu',
            input_shape=(*input_shape, ),
            data_format='channels_last',
        ),
        Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation='relu',
        ),
        Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
        ),
        Flatten(),
        Dense(
            units=512,
            activation='relu',
        ),
        Dense(
            units=n_actions,
        ),
    ])

    return model

def compile_model(model, lr=0.00025):
    optimizer = RMSprop(lr=lr)
    model.compile(
        optimizer=optimizer,
    )
if __name__ == "__main__":
    # input_shape = (84, 84, 4)
    # model = create_model(input_shape=input_shape)
    model = create_model()
    model.summary()
