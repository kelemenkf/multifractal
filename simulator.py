import numpy as np
import math
import matplotlib.pyplot as plt
from stochastic.processes.continuous import BrownianMotion
from stochastic.processes.continuous import FractionalBrownianMotion

from .multifractal import Multifractal


class Simulator():
    def __init__(self, sim_type='bm', T=1, n=100, H=0.5, loc=0, scale=1, drift=0):
        self.sim_type = sim_type
        self.T = T
        self.H = H
        self.loc = loc
        self.scale = scale
        self.k = math.ceil(math.log(self.T, 2))
        self.model = self.set_model()
        self.n = n
        self.drift = drift

    
    def get_increment(self, X):
        '''
        Returns the first difference of a vector. 
        '''
        return np.diff(X)

    #TODO add BM/FBM sampler.

    def sim_mmar(self):
        '''
        Simulates one path of a multifractal price series. If sim_type is 'mmar_m', 
        it simulates the martingale version using a standard Brwonian motion. If sim_type 
        is 'mmar' it simulates the MMAR using a fractional Brownian motion with an H supplied
        to the __init__ function. 
        '''
        #TODO check if this makes sense. 
        #TODO simulate at different time scales
        #TODO drift for FBM
        self.model.iterate(self.k)
        mu = self.model.get_measure()
        mu_increment = np.sqrt(mu)
        if self.sim_type == 'mmar_m':
            model = BrownianMotion(drift=self.drift, t=self.T)
        elif self.sim_type == 'mmar':
            model = FractionalBrownianMotion(hurst=self.H, t=self.T)
        s = model.sample(mu_increment.size)
        s = self.get_increment(s)
        self.model = self.set_model()
        return s*mu_increment
        

    def set_theta(self):
        '''
        Instantiates a multiplicative multifractal measure using lognormal multipliers,
        and loc and scale parameters, with support [0,T]. This is the model for trading time.  
        '''
        theta = Multifractal('lognormal', loc=self.loc, scale=self.scale, plot=False, support_endpoints=[0, self.T])
        return theta


    def set_model(self):
        '''
        Depending on sim_type, instantiates a model for simulation. 
        '''
        if self.sim_type == 'bm':
            b = BrownianMotion(t=self.T)
            return b
        elif self.sim_type == 'fbm':
            fbm = FractionalBrownianMotion(hurst=self.H, t=self.T)
            return fbm
        elif self.sim_type == 'mmar_m':
            return self.set_theta()
        elif self.sim_type == 'mmar':
            return self.set_theta()
    

    def plot_mmar(self):
        '''
        Plots a realization of a path of MMAR (cumulative returns). 
        '''
        y = self.sim_mmar()
        y = np.cumsum(y)
        x = range(y.size)
        plt.plot(x, y)

    
    def plot_mmar_lag(self):
        '''
        Plots the increments of a realization of a simulated MMAR path. 
        '''
        y = self.sim_mmar()
        x = range(y.size)
        plt.plot(x, y)
        if self.sim_type == 'mmar_m':
            plt.title("MMAR martingale")
        plt.xlabel('t')
        plt.ylabel('X(t)')


    def plot_bm(self):
        '''
        Plots the increments of a simple Brownian motion/Fractional Brownian Motion. 
        '''
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


    def plot_bm_diff(self):
        '''
        Plots the increments of a simple Brownian motion/Fractional Brownian Motion. 
        '''
        model = self.model
        y = model.sample(self.n)
        y = self.get_increment(y)
        x = range(y.size)
        plt.plot(x, y)
        

