import os
import time
import math
import random
from datetime import datetime

import numpy as np

if __name__ == "__main__":
    histLen = 4
    zero_out = False
    t = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]

    index = 0
    for i in range(1, histLen):
        idx = index + i
        if not zero_out:
            if t[idx] == 1:
                zero_out = True

        print(f't[{idx}]: {t[idx]} zero_out: {zero_out}')
