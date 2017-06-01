
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from code.plot.util.ggplot import GGPlot

observation_length = np.loadtxt(
    'hpc_asset'
    '/bytenet_small_europarl_nosummary_max500_adam_debug'
    '/observation_length.csv'
)

minibatch_length = observation_length.reshape(-1, 64).mean(axis=1)
s = pd.Series(minibatch_length[0:2000])
s.plot()

plt.show()
