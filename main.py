from functools import partial

import numpy as np
import logging
from func_gen import *
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.use("TkAgg")
plt.style.use('seaborn-whitegrid')

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, idx, x0: np.array, ctl):
        self.x = x0
        self.idx = idx
        self.n = x0.shape[0] // 2
        self.xmask = [self.idx * 2, self.idx * 2 + 1]
        self.counters = np.zeros(self.n)
        self.ctl = ctl
        # between 1 or 4 times delay
        self.clock = dict(delay=np.random.randint(1, 5), multiple=np.random.randint(1, 5))
        self.pgrad = partial(self.ctl.problem.penalty, i=self.idx)

    @property
    def pos(self):
        return self.x[self.xmask]

    @pos.setter
    def pos(self, val):
        self.x[self.xmask] = val

    def is_tick(self, i):
        return (i - self.clock['delay']) % self.clock['multiple'] == 0

    def step(self, i):
        if self.is_tick(i):
            # make gradient update
            logger.debug('%dth grad update of agent %d: %s', i, self.idx, self.ctl.fgrad(self.x)[1])
            self.pos -= self.ctl.lr * (
                    self.ctl.fgrad(self.x)[1][self.xmask] + self.ctl.beta * self.pgrad(self.x)[1][self.xmask])
            self.counters[self.idx] = i
            for i in [-1, 1]:
                to = self.idx + i
                if 0 <= to < self.n:
                    self.ctl.send(self.idx, to, (self.counters, self.x))

    def msg(self, data):
        cntrs, x = data
        mask = self.counters < cntrs
        self.x[np.repeat(mask, 2)] = x[np.repeat(mask, 2)]
        self.counters[mask] = cntrs[mask]
        logger.debug('%d: \t%s', self.idx, self.counters, self.x)


class Controller:
    """
    Will take care of network transmission probability between the agents and log all messages
    """

    def __init__(self, n, ideal_dist, problem):
        self.n = n
        self.ideal_dist = ideal_dist
        self.problem = problem(n, self.ideal_dist)
        self.fgrad = self.problem.objective

    def reset(self):
        x0 = np.random.uniform(size=2 * self.n)
        self.i = 0
        self.agents = [Agent(i, x0, self) for i in range(self.n)]
        return x0

    def step(self):
        for agent in self.agents:
            agent.step(self.i)
        self.i += 1
        return np.hstack([agent.pos for agent in self.agents])

    def send(self, src: int, to: int, msg):
        issucc, dist = self.check_channel(src, to)
        if issucc:
            self.agents[to].msg(msg)
            logger.debug('%d→%d: \t"%s" d=%.4f', src, to, msg, dist)

    def check_channel(self, src: int, to: int):
        """

        :param src: index of the sender agent
        :param to: index of the reciever agent
        :return:
        """
        src, to = self.agents[src], self.agents[to]
        dist = np.linalg.norm(src.pos - to.pos)

        # linear function with maximum at .4C and minimum at 1.5C (thats the zero position, but its actually clipped)
        return np.random.uniform() < np.clip((1.5 - dist / self.ideal_dist) / 1.1, .001, 1), dist

    @property
    def lr(self):
        return (3e-3)

    @property
    def beta(self):
        return 2 + self.i / 100


if __name__ == '__main__':
    ctrl = Controller(10, .1, LineProblem)

    logging.basicConfig(level=logging.WARN)

    fig, ax = plt.subplots()


    # ax.set_aspect('equal')

    def animate(i):
        if i == 0:
            x = ctrl.reset()
        else:
            x = ctrl.step()
        print(x)
        ax.clear()
        ax.plot(x[0::2].tolist(), x[1::2].tolist(), ':o')
        for agent in ctrl.agents:
            ax.annotate(re.sub('\. *', ':', str(i - agent.counters)[1:-2]), agent.pos, fontsize=8)
        return ax,


    ani = FuncAnimation(fig, animate, 2000, interval=10, repeat=False)
    # ani.save("out.mp4")
    plt.show()
