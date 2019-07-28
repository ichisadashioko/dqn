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
        maxSize=500_000,
    ):
        self.histLen = histLen
        self.maxSize = maxSize
        self.insertIndex = 0

        self.s = []
        self.a = []
        self.r = []
        self.t = []

    def __len__(self):
        return min((len(self.s), len(self.a), len(self.r), len(self.t)))

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
