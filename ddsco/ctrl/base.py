import logging

import numpy as np

from ddsco.agent import Agent

from ddsco.agent import LocalObjAgent

logger = logging.getLogger(__name__)


class Controller:
    """
    Will take care of network transmission probability between the agents and log all messages
    """

    def __init__(self, n, ideal_dist, problem, link_quality=(.4, 1.5)):
        self.n = n
        self.ideal_dist = ideal_dist
        self.problem = problem(n, self.ideal_dist)
        self.fgrad = self.problem.objective
        self.link_quality = link_quality

        self.agents = None
        self.i = None

    def reset(self, x0=None):
        if x0 is None:
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
        return np.random.uniform() < np.clip(
            (self.link_quality[1] - dist / self.ideal_dist) / (self.link_quality[1] - self.link_quality[0]), .01,
            1), dist

    @property
    def lr(self):
        return 100 / (self.i + 30000)

    @property
    def beta(self):
        return 1 + (self.i / 100) ** .5


class LocalObjCtrl(Controller):
    def reset(self, x0=None):
        if x0 is None:
            x0 = np.tile(np.random.uniform(size=2), self.n)
        self.i = 0
        self.agents = [LocalObjAgent(i, x0, self) for i in range(self.n)]
        return x0
