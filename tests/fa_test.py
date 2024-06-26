import numpy as np

from repos.multifractal.time_series.fluctuation_analysis import FluctuationAnalysis
from repos.multifractal.time_series.simulator import Simulator


i = 0
alphas = []
while i < 100:
  sim = Simulator('bm', T=65536, dt_scale=1)
  sample = sim.sim_bm(65536)
  fa = FluctuationAnalysis(sample[0])
  alphas.append(fa.alpha)


print(np.mean(alphas))


