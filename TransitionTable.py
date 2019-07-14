import os
import time
import math
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


class TransitionTable:
    def __init__(
        self,
        stateDim=(105, 80),
        histLen=1,
        maxSize=1_000_000,
        bufferSize=1024,
    ):
        self.stateDim = stateDim
        self.histLen = histLen
        self.maxSize = maxSize
        self.bufferSize = bufferSize
        self.buf_ind = None

        self.recentMemSize = self.histLen

        self.numEntries = 0
        self.insertIndex = 0

        # The original implementation has multiple `histType`, we are going to use 'linear' only. Because of that, there is no `histIndices`

        # DONE pre-allocate (maxSize, dims) Tensors
        self.s = np.zeros(shape=(self.maxSize, *self.stateDim), dtype=np.uint8)
        self.a = np.zeros(self.maxSize, dtype=np.uint8)
        self.r = np.zeros(self.maxSize, dtype=np.float32)
        self.t = np.zeros(self.maxSize, dtype=np.uint8)

        # Tables for storing the last `histLen` states. They are used for constructing the most recent agent state more easily
        self.recent_s = []
        self.recent_a = []
        self.recent_t = []

        # DONE pre-allocate Tensors
        s_size = (self.histLen, *self.stateDim)  # TODO 3 consider between 'channels_first' or 'channels_last'
        self.buf_a = np.zeros(self.bufferSize, dtype=np.uint8)
        self.buf_r = np.zeros(self.bufferSize, dtype=np.float32)
        self.buf_term = np.zeros(self.bufferSize, dtype=np.uint8)
        # TODO 4 check the buffer shape before pass it to the model
        self.buf_s = np.zeros(s_size, dtype=np.uint8)
        self.buf_s2 = np.zeros(s_size, dtype=np.uint8)

    def reset(self):  # DONE
        self.numEntries = 0
        self.insertIndex = 0

    def size(self):  # DONE
        return self.numEntries

    def empty(self):  # DONE
        return self.numEntries == 0

    def fill_buffer(self):  # TODO 3
        assert self.numEntries >= self.bufferSize
        # clear CPU buffers
        self.buf_ind = 1

        for buf_ind in range(self.bufferSize):
            s, a, r, s2, term = self.sample_one()
            self.buf_s[buf_ind] = s
            self.buf_a[buf_ind] = a
            self.buf_r[buf_ind] = r
            self.buf_s2[buf_ind] = s2
            self.buf_term[buf_ind] = term

        self.buf_s = self.buf_s / 255.0
        self.buf_s2 = self.buf_s2 / 255.0

    def sample_one(self):  # TODO 3
        assert self.numEntries > 1

        valid = False
        while not valid:
            # start at the second index because of previous action
            index = random.randrange(1, self.numEntries - self.recentMemSize)

            if self.t[index + self.recentMemSize - 1] == 0:
                valid = True

        return self.get(index)

    def sample(self, batch_size=1):  # TODO 4
        assert batch_size < self.bufferSize

        if (self.buf_ind is None) or (self.buf_ind + batch_size) > self.bufferSize:
            self.fill_buffer()

        index = self.buf_ind

    def concatFrames(self, index, use_recent=False):  # DONE 4
        """
        The `index` must not be the terminal state
        """
        if use_recent:
            s, t = self.recent_s, self.recent_t
        else:
            s, t = self.s, self.t

        # DONE copy frames and zeros pad missing frames
        fullstate = np.zeros(shape=(self.histLen, *self.stateDim), dtype=np.uint8)

        end_index = min(len(s) - 1, index + self.histLen)

        for fs_idx, i in enumerate(range(index, end_index)):
            fullstate[fs_idx] = np.copy(s[i])

        # DONE 5 copy frames and zero-out un-related frames
        # Because all the episode frames is stack together, the below code is use to find the terminal state index (episode-seperator) and zero out all the frames after that index.
        zero_out = False

        for i in range(1, self.histLen):
            if not zero_out:
                idx = index + i
                if t[idx] == 1:
                    zero_out = True

            if zero_out:
                fullstate[i] = np.zeros_like(fullstate[i].shape)

        return fullstate

    def concatActions(self, index, use_recent=False):  # TODO 4
        act_hist = []
        if use_recent:
            a, t = self.recent_a, self.recent_t
        else:
            a, t = self.a, self.t

        # zero out frames from all but the most recent episode
        # TODO 5

    def get_recent(self):  # DONE
        # Assumes that the most recent state has been added, but the action has not
        return self.concatFrames(0, True)

    def get(self, index):  # DONE
        s = self.concatFrames(index)
        s2 = self.concatFrames(index + 1)
        ar_index = index + self.recentMemSize - 1

        return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index + 1]

    def add(self, s, a, r, term):  # DONE
        # Increment until at full capacity
        if self.numEntries < self.maxSize:
            self.numEntries += 1

        # Always insert at next index, then wrap around
        self.insertIndex += 1
        # Overwrite oldest experience once at capacity
        if self.insertIndex >= self.maxSize:
            self.insertIndex = 0

        # Overwrite (s, a, r, t) at `insertIndex`
        self.s[self.insertIndex] = s
        self.a[self.insertIndex] = a
        self.r[self.insertIndex] = r
        if term:
            self.t[self.insertIndex] = 1
        else:
            self.t[self.insertIndex] = 0

    def add_recent_state(self, s, term):  # TODO
        if len(self.recent_s) == 0:
            for i in range(self.recentMemSize):
                pass

    def add_recent_action(self, a):  # TODO
        pass
