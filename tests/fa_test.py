import sys
import os
import argparse 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


parser = argparse.ArgumentParser(description="Tests the FluctuationAnalysis class.")
parser.add_argument("N_samples", type=int, metavar="N", help="The number of paths to be simulated")
args = parser.parse_args()


import numpy as np

from multifractal.time_series.fluctuation_analysis import FluctuationAnalysis
from multifractal.simulator import Simulator


i = 0
samples = args.N_samples
alphas = []

while i < samples:
  sim = Simulator('bm', T=65536, dt_scale=1)
  sample = sim.sim_bm(65536)
  fa = FluctuationAnalysis(sample[0])
  print(fa.alpha)
  alphas.append(fa.alpha)
  i += 1


print(f"Mean alpha based on {samples} simulated paths: {np.mean(alphas)}")


