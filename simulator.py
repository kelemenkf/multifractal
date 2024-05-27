import numpy as np
import math
import matplotlib.pyplot as plt
from stochastic.processes.continuous import BrownianMotion
from stochastic.processes.continuous import FractionalBrownianMotion

from .multifractal import Multifractal


class Simulator():
    def __init__(self, sim_type='bm', T=1, n=100, H=0.5, loc=0, scale=1, drift=False):
        self.sim_type = sim_type
        self.T = T
        self.k = math.ceil(math.log(self.T, 2)) 
        self.H = H
        self.loc = loc
        self.scale = scale
        self.subordinator = self.set_subordinator()
        if self.sim_type == 'mmar_r' or self.sim_type == 'mmar':
            self.T = self.subordinator.b**self.k
        self.drift = drift
        self.subordinated = self.set_subordinated()
        self.n = n

    
    def get_increment(self, X):
        '''
        Returns the first difference of a vector. 
        '''
        return np.diff(X)

    def sim_bm(self, n):
        times = self.subordinated.times(n)
        if self.sim_type == 'bm':
            return (self.subordinated.sample(n), times)
        elif self.sim_type == 'fbm':
            return (self.subordinated.sample(n), times)


    def sim_mmar(self):
        '''
        Simulates one path of a multifractal price series. If sim_type is 'mmar_m', 
        it simulates the martingale version using a standard Brwonian motion. If sim_type 
        is 'mmar' it simulates the MMAR using a fractional Brownian motion with an H supplied
        to the __init__ function. 
        '''
        #TODO check if this makes sense in the case of FBM. 
        #TODO simulate at different time scales
        #TODO drift for FBM
        #TODO README for the reason for the number of iterations. 
        self.subordinator.iterate(self.k)
        mu = self.subordinator.get_measure()
        mu_increment = np.sqrt(mu)

        mu_increment_size = mu_increment.size

        if self.sim_type == 'mmar_m':
            self.sim_type = 'bm'
        elif self.sim_type == 'mmar':
            self.sim_type = 'fbm'

        s, times = self.sim_bm(mu_increment_size)
        s = self.get_increment(s)

        #Reset it because an instance of Multifractal keeps track of k. 
        self.subordinator = self.set_subordinator()

        return (s*mu_increment, times)
    

    def set_subordinator(self):
        '''
        Instantiates a multiplicative multifractal measure using lognormal multipliers,
        and loc and scale parameters, with support [0,T]. This is the model for trading time.  
        '''

        theta = Multifractal('lognormal', loc=self.loc, scale=self.scale, plot=False, support_endpoints=[0, self.T])
        return theta


    def set_subordinated(self):
        if self.sim_type == 'mmar_m' or 'bm':
            return BrownianMotion(drift=self.drift, t=self.T)
        elif self.sim_type == 'mmar' or 'fbm':
            return FractionalBrownianMotion(hurst=self.H, t=self.T)


    def plot_mmar(self):
        '''
        Plots a realization of a path of MMAR (cumulative returns). 
        '''
        y, _ = self.sim_mmar()
        y = np.cumsum(y)
        x = range(y.size)
        plt.plot(x, y)

    
    def plot_mmar_diff(self):
        '''
        Plots the increments of a realization of a simulated MMAR path. 
        '''
        y, x = self.sim_mmar()
        plt.plot(x[1:], y)
        if self.sim_type == 'mmar_m':
            plt.title("MMAR martingale")
        plt.xlabel('t')
        plt.ylabel('X(t)')


    def plot_bm(self):
        '''
        Plots the increments of a simple Brownian motion/Fractional Brownian Motion. 
        '''
        y, x = self.sim_bm(self.n)
        if self.sim_type == 'bm':
            plt.title('Brownian Motion')
        elif self.sim_type == 'fbm':
            plt.title('Fractional Brownian Motion')
        plt.plot(x, y)
        plt.xlabel('t')
        plt.ylabel('W(t)')


    def plot_bm_diff(self):
        '''
        Plots the increments of a simple Brownian motion/Fractional Brownian Motion. 
        '''
        y, x = self.sim_bm(self.n)
        y = self.get_increment(y)
        plt.plot(x[1:], y)
        plt.xlabel('t') 
        plt.ylabel('X(t)')
