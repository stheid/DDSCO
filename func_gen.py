from typing import Tuple
import numpy as np
import torch
from matplotlib.patches import Circle
from torch.distributions import normal

torch.manual_seed(1)
np.random.seed(1)


class Problem:
    def __init__(self, n, ideal_dist):
        self.n = n
        self.ideal_dist = ideal_dist

    def objective(self, x) -> Tuple[float, np.array]:
        """

        :param x: input value for the function
        :return: value and nabla at x
        """
        pass

    def penalty(self, i, x) -> Tuple[float, np.array]:
        """
        create penalties of the ith robot with n robots in total
        :param i: ith robot
        :param x: current value
        :return: value and nabla at x
        """
        pass

    def draw(self, x, ax):
        pass


class LineProblem(Problem):
    def _reg(self, x, is_noisy=True):
        src = torch.tensor(x, dtype=torch.double, requires_grad=True)
        pdf = normal.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([1.]))
        noise = pdf.sample(torch.Size([self.n]))

        # split src vector into x and y
        x, y = src.reshape((-1, 2)).unsqueeze(1).unbind(2)
        # add noise and bias value
        cols = [torch.ones_like(x)] + ([noise] if is_noisy else []) + [x]
        x = torch.cat(cols, 1)
        # solve OLS
        b = torch.inverse(x.T @ x) @ x.T @ y
        return src, x, y, b

    def objective(self, x):
        src, x, y, b = self._reg(x)
        # estimate regressionline
        y_ = x @ b
        # calculate residual error
        e = y - y_
        # calculate dist between first and last
        d = torch.norm(src[:2] - src[-2:])
        # minimize error and maximize dist
        loss = e.T @ e - d / self.n

        src.retain_grad()
        loss.backward()
        return loss.detach().numpy(), src.grad.numpy()

    def penalty(self, x, i):
        """
        distance to between neightbour must be a constant.
        :param x:
        :param i:
        :return:
        """
        src = torch.tensor(x, dtype=torch.double, requires_grad=True)

        g = torch.zeros(2)
        for idx, j in enumerate([-1, 1]):
            j = i + j
            if 0 <= j < self.n:
                # but neighbours should also not exceed C
                g[idx] = torch.abs(self.ideal_dist - torch.norm(src[i * 2:i * 2 + 2] - src[j * 2:j * 2 + 2]))
        loss = torch.relu(g).pow(2).sum()

        src.retain_grad()
        loss.backward()
        return loss.detach().numpy(), src.grad.numpy()

    def draw(self, x, ax):
        # only use b, convert to numpy, and discard the noise term
        b = self._reg(x)[3].detach().numpy()[[0, 2]]
        x = np.array([ax.get_xlim()])
        y = np.hstack((np.ones_like(x.T), x.T)) @ b

        ax.plot(x.flatten(), y.flatten(), 'k', alpha=.5, lw=1)


class PerpLineProblem(LineProblem):
    def objective(self, x):
        src, x, y, b = self._reg(x, is_noisy=False)
        if np.random.randint(2) == 0:
            # calculate perpendicular line at origin
            b[1] = -1 / b[1]
        # estimate regressionline
        y_ = x @ b
        # calculate residual error
        e = y - y_
        # calculate dist between first and last
        d = torch.norm(src[:2] - src[-2:])
        # minimize error and maximize dist
        loss = e.T @ e - d / self.n

        src.retain_grad()
        loss.backward()
        return loss.detach().numpy(), src.grad.numpy()

    def draw(self, x, ax):
        # only use b, convert to numpy
        b = self._reg(x, False)[3].detach().numpy()

        x = np.array([ax.get_xlim()])
        _x = np.hstack((np.ones_like(x.T), x.T))

        ax.plot(x.flatten(), (_x @ b).flatten(), 'k', alpha=.5, lw=1)
        b[1] = -1 / b[1]
        ax.plot(x.flatten(), (_x @ b).flatten(), 'k:', alpha=.5, lw=1)


class CircleProblem(Problem):
    def objective(self, x) -> Tuple[float, np.array]:
        src = torch.tensor(x, dtype=torch.double, requires_grad=True)
        x, y = src.reshape((-1, 2)).unsqueeze(1).unbind(2)

        a = torch.zeros(self.n)
        for i in range(self.n):
            next = (i + 1) % self.n
            a[i] = x[i] * y[next] - x[next] * y[i]
        # maximize the sum of all terms
        loss = -torch.abs(a.sum())

        src.retain_grad()
        loss.backward()
        return loss.detach().numpy(), src.grad.numpy()

    def penalty(self, x, i):
        """
        distance to between neightbour must be a constant. 0 and n-1 are also considered neighbours
        :param x:
        :param i:
        :return:
        """

        src = torch.tensor(x, dtype=torch.double, requires_grad=True)

        g = torch.zeros(2)
        for idx, j in enumerate([-1, 1]):
            j = (i + j) % self.n
            # but neighbours should also not exceed C
            g[idx] = torch.abs(self.ideal_dist - torch.norm(src[i * 2:i * 2 + 2] - src[j * 2:j * 2 + 2]))
        loss = torch.relu(g).pow(2).sum()

        src.retain_grad()
        loss.backward()
        return loss.detach().numpy(), src.grad.numpy()

    def draw(self, x, ax):
        """
        The drawn circle might look undersized, but this is not true. The agents force outside in order to maximize area.
        Only the constraint is holding them on the circle. Therefore, with a growing penalty term, the agents will eventually line up directly on the circle.
        :param x:
        :param ax:
        :return:
        """
        xy = x.reshape((-1, 2)).mean(axis=0)
        # radius of a n-sided regular polygon
        circ = Circle(xy, self.ideal_dist / (2 * np.sin(np.pi / self.n)), fill=False)
        ax.add_patch(circ)
