import logging
from functools import partial
import numpy as np

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, idx, x0: np.array, ctl):
        self.x = x0.copy()
        self.idx = idx
        self.n = x0.shape[0] // 2
        self.xmask = [self.idx * 2, self.idx * 2 + 1]
        self.counters = np.zeros(self.n)
        self.ctl = ctl
        # between 1 or 4 times delay
        self.clock = dict(delay=np.random.randint(3, 7), multiple=np.random.randint(3, 7))
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
            self.pos -= self.ctl.lr * (
                    self.ctl.fgrad(self.x)[1][self.xmask] + self.ctl.beta * self.pgrad(self.x)[1][self.xmask])
            logger.debug('%dth grad update of agent %d: %s', i, self.idx, self.x)
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
