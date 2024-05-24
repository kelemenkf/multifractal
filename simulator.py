import numpy as np
import math
import matplotlib.pyplot as plt
from stochastic.processes.continuous import BrownianMotion
from stochastic.processes.continuous import FractionalBrownianMotion

from .multifractal import Multifractal


class Simulator():
    def __init__(self, sim_type='bm', t=1, n=100, H=0.5, loc=0, scale=1):
        self.sim_type = sim_type
        self.t = t
        self.H = H
        self.loc = loc
        self.scale = scale
        self.k = math.ceil(math.log(self.t, 2))
        self.model = self.set_model()
        self.n = n


    def sim_mmar(self):
        if self.sim_type == 'mmar_m':
            theta = Multifractal('lognormal', loc=self.loc, scale=self.scale, plot=False)
            theta.iterate(self.k)
            mu = theta.get_measure()



    def set_model(self):
        if self.sim_type == 'bm':
            b = BrownianMotion(t=self.t)
            return b
        elif self.sim_type == 'fbm':
            fbm = FractionalBrownianMotion(hurst=self.H, t=self.t)
            return fbm
        elif self.sim_type == 'mmar_m':
            self.sim_mmar()
            

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
        

