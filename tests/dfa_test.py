import sys
import os
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="Tests the DetrendedFluctuationAnalysis class")
parser.add_argument("N_samples", type=int, metavar="N", help="The number of paths to be simulated")
args = parser.parse_args()

import numpy as np

from multifractal.src.time_series.detrended_fluctuation_analysis import DFA
from multifractal.src.simulator import Simulator

i = 0
samples = args.N_samples
alphas = [[], []]


print("Wiener Brownian Motion:")
while i < samples:
  sim = Simulator('bm', T=65536, dt_scale=1)
  sample = sim.sim_bm(65536)
  dfa = DFA(sample[0])
  print(dfa.alpha)
  alphas[0].append(dfa.alpha)
  i += 1

i = 0

print("Fractional Brownian Motion:")
while i < samples: 
  sim = Simulator('fbm', T=65536, dt_scale=1, H=0.7)
  sample = sim.sim_bm(65536)
  dfa = DFA(sample[0])
  print(dfa.alpha)
  alphas[1].append(dfa.alpha)
  i += 1


print(f"Mean alpha based on {samples} simulated paths of WBM: {np.mean(alphas[0])}")
print(f"Mean alpha based on {samples} simulated paths of FBM with H=0.7: {np.mean(alphas[1])}")


