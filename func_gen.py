from typing import Tuple
import numpy as np
import torch
from torch.distributions import normal


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


class LineProblem(Problem):

    def objective(self, x):
        src = torch.tensor(x, dtype=torch.double, requires_grad=True)
        pdf = normal.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([1.]))
        noise = pdf.sample(torch.Size([self.n]))

        # split src vector into x and y
        x, y = src.reshape((-1, 2)).unsqueeze(1).unbind(2)
        # add noise and bias value
        x = torch.cat((torch.ones_like(x), noise, x), 1)
        # solve OLS
        b = torch.inverse(x.T @ x) @ x.T @ y
        # estimate regressionline
        y_ = x @ b
        # calculate residual error
        e = y - y_
        # calculate dist between first and last
        d = torch.norm(src[:2] - src[-2:])
        # minimize error and maximize dist
        loss = e.T @ e - d

        src.retain_grad()
        loss.backward()
        return loss.detach().numpy(), src.grad.numpy()

    def penalty(self, x, i):
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
