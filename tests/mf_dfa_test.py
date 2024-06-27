import sys
import os
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="Tests the MF_DFA class")
parser.add_argument("N_samples", type=int, metavar="N", help="The number of paths to be simulated")
args = parser.parse_args()

import numpy as np
from multifractal.src.time_series.mf_dfa import MF_DFA
from multifractal.src.simulator import Simulator

i = 0
samples = args.N_samples
H = []


print("Long-range uncorrelated series (Wiener Brownian Motion)")
while i < samples:
  sim = Simulator('bm', T=65536, dt_scale=1)
  sample = sim.sim_bm(65536)
  mf_dfa = MF_DFA(sample[0], modified=False, data_type="profile", nu_max=10)
  print(f"Mean: {np.mean(list(mf_dfa.h_q.values()))}, rang: {min(mf_dfa.h_q.values())}, {max(mf_dfa.h_q.values())})")
  h_mean = np.mean(list(mf_dfa.h_q.values()))
  H.append(h_mean)
  i += 1



