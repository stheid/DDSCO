import logging

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.animation import FuncAnimation

from ddsco.ctrl import Controller, LocalObjCtrl
from ddsco.problems import *

logger = logging.getLogger(__name__)

matplotlib.use("TkAgg")
plt.style.use('seaborn-whitegrid')

torch.manual_seed(1)
np.random.seed(1)

if __name__ == '__main__':
    # ctrl = Controller(10, .1, CircleProblem, link_quality=[-2, 1.1])
    ctrl = Controller(10, .1, LineProblem, link_quality=[-2, 1])
    # ctrl = LocalObjCtrl(4, .1, LocalObjectiveSharedX, link_quality=[-500000, 1])

    logging.basicConfig(level=logging.WARN)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal', 'box')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    def animate(i, speedup=100):
        if i == 0:
            # x = ctrl.reset(np.array([0., 0., .1, 0., .1, .1, .2, .1]))
            x = ctrl.reset()
        else:
            for _ in range(speedup):
                x = ctrl.step()
        print(i * speedup, ctrl.problem.objective(x)[0].item())
        ax.clear()
        # ax.set_title(ctrl.link_quality)
        ax.plot(x[0::2].tolist(), x[1::2].tolist(), ':o')
        for agent in ctrl.agents:
            # ax.annotate(':'.join([f'{int(c):d}' for c in (i * speedup) - agent.counters]), agent.pos, fontsize=8)
            if hasattr(agent, 'min'):
                ax.plot(*agent.min, 'kx')
        ax.autoscale_view()
        ax.autoscale(False)
        ctrl.problem.draw(x, ax)
        return ax,


    ani = FuncAnimation(fig, animate, 1000, interval=10, repeat=False)
    ani.save(f'../results/line-fast{ctrl.link_quality}_nonumbers.mp4')
    # plt.show()

# ffmpeg -i circle\[0\,2\].mp4 -i circle\[0\,1.1\].mp4 -filter_complex hstack output.mp4
