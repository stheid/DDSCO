import logging

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from ddsco import Controller
from ddsco.problems import *

logger = logging.getLogger(__name__)

plt.style.use('seaborn-whitegrid')

torch.manual_seed(1)
np.random.seed(1)

if __name__ == '__main__':
    for problem in [CircleProblem]:
        losses = {}
        penalties = {}
        for quali in [1.1, 3]:
            ctrl = Controller(10, .1, problem, link_quality=[0, quali])

            x = ctrl.reset()
            runs = []
            runs2 = []
            for _ in range(10 if problem == LineProblem else 1):
                loss = [ctrl.problem.objective(x)[0].item()]
                penal = [sum([a.pgrad(x)[0].item() for a in ctrl.agents])]

                for i in range(40000):
                    x = ctrl.step()
                    l = ctrl.problem.objective(x)[0].item()
                    p = sum([a.pgrad(x)[0].item() for a in ctrl.agents])
                    loss.append(l)
                    penal.append(p)
                runs.append(np.array(loss))
                runs2.append(np.array(penal))

            losses[quali] = np.vstack(runs).mean(axis=0)
            penalties[quali] = np.vstack(runs2).mean(axis=0)
        df = pd.DataFrame(losses)
        df.plot()
        plt.show()
        df = pd.DataFrame(penalties)
        df.plot()
        plt.show()
