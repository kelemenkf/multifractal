import numpy as np
import matplotlib.pyplot as plt
from stochastic.processes.continuous import BrownianMotion
from stochastic.processes.continuous import FractionalBrownianMotion


class Simulator():
    def __init__(self, sim_type='bm', t=1, n=100, H=0.5):

        self.sim_type = sim_type
        self.t = t
        self.H = H
        self.model = self.set_model()
        self.n = n


    def set_model(self):
        if self.sim_type == 'bm':
            b = BrownianMotion(t=self.t)
            return b
        elif self.sim_type == 'fbm':
            fbm = FractionalBrownianMotion(hurst=self.H, t=self.t)
            return fbm
            

    def plot_bm(self):
        model = self.model
        y = model.sample(self.n)
        x = model.times(self.n)
        if self.sim_type == 'bm':
            plt.title('Brownian Motion')
        elif self.sim_type == 'fbm':
            plt.title('Fractional Brownian Motion')
        plt.plot(x, y)
        plt.xlabel('t')
        plt.ylabel('X(t)')
        

