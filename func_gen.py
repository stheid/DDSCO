import torch
from torch.distributions import uniform


def objective(n):
    """

    :param n: agents
    :param terms: number of random quadratic functions
    :return:
    """

    def f(x):
        src = torch.tensor(x, dtype=torch.double, requires_grad=True)
        pdf = uniform.Uniform(torch.DoubleTensor([-1.]), torch.DoubleTensor([1.]))
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

        g = torch.zeros(n)
        for j in range(n):
            if i == j:
                continue

            # all non-neighbours should also be at least C away
            v = C - torch.norm(src[i * 2:i * 2 + 2] - src[j * 2:j * 2 + 2])
            if abs(i - j) == 1:
                # but neighbours should also not exceed C
                v = torch.abs(v)
            g[j] = v
        loss = torch.relu(g).pow(2).sum()

        src.retain_grad()
        loss.backward()
        return loss.detach().numpy(), src.grad.numpy()

    return p


if __name__ == '__main__':
    p = penalty(3, 0, 1)
    print(p([0, 0, 0, 1.5, 0, 2.5]))
