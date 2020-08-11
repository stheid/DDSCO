import numpy as np
import logging
from func_gen import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.use("TkAgg")
plt.style.use('seaborn-whitegrid')
C = .1


def lr(i):
    return (3e-3)


def beta(i):
    return 2 + i / 100
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, idx, x0: np.array, net):
        self.x = x0
        self.idx = idx
        self.n = x0.shape[0] // 2
        self.xmask = [self.idx * 2, self.idx * 2 + 1]
        self.counters = np.zeros(self.n)
        self.net = net
        # between 1 or 4 times delay
        self.clock = dict(delay=np.random.randint(1, 5), multiple=np.random.randint(1, 5))
        self.pgrad = penalty(self.n, self.idx, C)

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
            logger.debug('%dth grad update of agent %d: %s', i, self.idx, self.net.fgrad(self.x)[1])
            self.pos -= lr(i) * (self.net.fgrad(self.x)[1][self.xmask] + beta(i) * self.pgrad(self.x)[1][self.xmask])

            for i in [-1, 1]:
                to = self.idx + i
                if 0 <= to < self.n:
                    self.net.send(self.idx, to, (self.counters, self.x))

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

    def __init__(self, n):
        self.n = n
        self.fgrad = objective(n)

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
        return np.random.uniform() < np.clip((1.5 - dist / C) / 1.1, .001, 1), dist


if __name__ == '__main__':
    ctrl = Controller(10)

    fig, ax = plt.subplots()


    def animate(i):
        if i == 0:
            x = ctrl.reset()
        else:
            x = ctrl.step()
        print(x)
        ax.clear()
        ax.plot(x[0::2].tolist(), x[1::2].tolist(), ':o')
        return ax,


    ani = FuncAnimation(fig, animate, 2000, interval=10, repeat=False)
    # ani.save("out.mp4")
    plt.show()
