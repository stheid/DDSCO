import logging

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ctrl import Controller
from func_gen import *

logger = logging.getLogger(__name__)

matplotlib.use("TkAgg")
plt.style.use('seaborn-whitegrid')

if __name__ == '__main__':
    ctrl = Controller(4, .1, PerpLineProblem, link_quality=[0, 5000])

    logging.basicConfig(level=logging.WARN)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal', 'box')


    def animate(i):
        if i == 0:
            # x = ctrl.reset(np.array([0., 0., .1, 0., .1, .1, .2, .1]))
            x = ctrl.reset()
        else:
            x = ctrl.step()
        logger.info(x)
        ax.clear()
        ax.set_title(ctrl.link_quality)
        ax.plot(x[0::2].tolist(), x[1::2].tolist(), ':o')
        ax.autoscale_view()
        ax.autoscale(False)
        ctrl.problem.draw(x, ax)
        for agent in ctrl.agents:
            ax.annotate(':'.join([f'{int(c):d}' for c in i - agent.counters]), agent.pos, fontsize=8)
        return ax,


    ani = FuncAnimation(fig, animate, 50000, interval=10, repeat=False)
    ani.save(f'perp-zigzaginit{ctrl.link_quality}.mp4')
    # plt.show()

# ffmpeg -i circle\[0\,2\].mp4 -i circle\[0\,1.1\].mp4 -filter_complex hstack output.mp4
