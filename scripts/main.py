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

torch.manual_seed(2)
np.random.seed(2)

if __name__ == '__main__':
    # ctrl = Controller(4, .1, PerpLineProblem, link_quality=[0, 5000])
    ctrl = LocalObjCtrl(4, .1, LocalObjectiveSharedX, link_quality=[-500000, 1])

    logging.basicConfig(level=logging.WARN)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal', 'box')


    def animate(i, speedup=1):
        if i == 0:
            # x = ctrl.reset(np.array([0., 0., .1, 0., .1, .1, .2, .1]))
            x = ctrl.reset()
        else:
            for i in range(speedup):
                x = ctrl.step()
        logger.info(x)
        ax.clear()
        ax.set_title(ctrl.link_quality)
        ax.plot(x[0::2].tolist(), x[1::2].tolist(), ':o')
        for agent in ctrl.agents:
            ax.annotate(':'.join([f'{int(c):d}' for c in i * speedup - agent.counters]), agent.pos, fontsize=8)
            if hasattr(agent, 'min'):
                ax.plot(*agent.min, 'kx')
        ax.autoscale_view()
        ax.autoscale(False)
        ctrl.problem.draw(x, ax)
        return ax,


    ani = FuncAnimation(fig, animate, 5000000, interval=10, repeat=False)
    # ani.save(f'results/perp-randinit-fast{ctrl.link_quality}.mp4')
    plt.show()

# ffmpeg -i circle\[0\,2\].mp4 -i circle\[0\,1.1\].mp4 -filter_complex hstack output.mp4
