import torch
from torch.distributions import normal


def objective(n):
    """

    :param n: agents
    :param terms: number of random quadratic functions
    :return:
    """

    def f(x):
        src = torch.tensor(x, dtype=torch.double, requires_grad=True)
        pdf = normal.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([1.]))
        noise = pdf.sample(torch.Size([n]))

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

    return f


def penalty(n, i, C):
    """
    create penalties of the ith robot with n robots in total
    :param n: number of robots
    :param i: ith robot
    :return:
    """

    def p(x):
        src = torch.tensor(x, dtype=torch.double, requires_grad=True)

        g = torch.zeros(2)
        for idx, j in enumerate([-1, 1]):
            j = i + j
            if 0 <= j < n:
                # but neighbours should also not exceed C
                g[idx] = torch.abs(C - torch.norm(src[i * 2:i * 2 + 2] - src[j * 2:j * 2 + 2]))
        loss = torch.relu(g).pow(2).sum()

        src.retain_grad()
        loss.backward()
        return loss.detach().numpy(), src.grad.numpy()

    return p


if __name__ == '__main__':
    p = penalty(3, 0, 1)
    print(p([0, 0, 0, 1.5, 0, 2.5]))
