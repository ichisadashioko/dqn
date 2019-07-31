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
        histLen=4,
        stateDim=(105, 80),
        maxSize=500_000,
    ):
        self.histLen = histLen
        self.stateDim = stateDim
        self.maxSize = maxSize
        self.insertIndex = 0

        self.s = [] # state
        self.a = [] # action
        self.r = [] # reward
        self.t = [] # terminal

        # use recent buffer to get the latest state.
        # recent buffer is used by the agent
        # when the agent perceive the most recent state
        self.recentMemSize = self.histLen
        self.recent_s = []
        self.recent_t = []

    def __len__(self):
        return min((
            len(self.s),
            len(self.a),
            len(self.r),
            len(self.t),
        ))

    def add(self, s, a, r, t):
        while len(self.s) < self.maxSize:
            self.s.append(None)
            self.a.append(None)
            self.r.append(None)
            self.t.append(None)

        self.insertIndex %= self.maxSize
        self.s[self.insertIndex] = s
        self.a[self.insertIndex] = a
        self.r[self.insertIndex] = r
        self.t[self.insertIndex] = t

        self.insertIndex += 1

    def add_recent_state(self, s, t):
        while len(self.recent_s) < self.recentMemSize:
            self.recent_s.append(np.zeros(shape=self.stateDim), dtype=np.uint8)
            self.recent_t.append(0)
        
        self.recent_s.append(s)
        self.recent_t.append(t)
        

    def concatFrames(self, index, recent=None):
        if recent:
            pass

    def get_recent(self):
        pass
