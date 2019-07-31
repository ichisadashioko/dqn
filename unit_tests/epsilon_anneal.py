import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class EpsilonAnnealing:
    def __init__(
        self,
        ep_start=1,
        ep_end=0.1,
        ep_endt=1_000_000,
        learn_start=10_000,
    ):
        self.numSteps = 0
        self.ep_start = ep_start
        self.ep_end = ep_end
        self.ep_endt = ep_endt
        self.learn_start = learn_start

    def calc_epsilon(self):
        # retval = (self.ep_start - self.ep_end)
        # retval = retval * ((self.ep_endt - self.numSteps) / self.ep_endt)
        # retval = max(retval, self.ep_end)

        ep_range = self.ep_start - self.ep_end
        ep_prog = 1.0 - max(0, self.numSteps - self.learn_start) / self.ep_endt
        ep_delta = ep_range * ep_prog
        retval = self.ep_end + max(0, ep_delta)

        return retval

    def step(self):
        self.numSteps += 1


if __name__ == "__main__":
    epsTestSub = EpsilonAnnealing()

    eps_log = []
    num_steps = 2_000_000
    for _ in tqdm(range(num_steps)):
        epsTestSub.step()

        eps = epsTestSub.calc_epsilon()
        eps_log.append(eps)

    eps_log = np.array(eps_log)
    plt.plot(eps_log)
    plt.show()
