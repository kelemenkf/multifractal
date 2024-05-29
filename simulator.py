import numpy as np
import math
import matplotlib.pyplot as plt
from stochastic.processes.continuous import BrownianMotion
from stochastic.processes.continuous import FractionalBrownianMotion

from .multifractal import Multifractal


class Simulator():
    def __init__(self, sim_type='bm', T=1000, dt_scale=1, H=0.5, loc=0, scale=1, drift=0, diffusion=1):
        self.sim_type = sim_type
        self.T = T
        self.dt_scale = dt_scale
        self.k = math.ceil(math.log(self.T, 2)) 
        self.H = H
        self.loc = loc
        self.scale = scale
        self.subordinator = self.set_subordinator()
        if self.sim_type == 'mmar_r' or self.sim_type == 'mmar':
            self.T = self.subordinator.b**self.k
        self.drift = drift
        self.diffusion = diffusion
        self.subordinated = self.set_subordinated()
        self.n = self.T // self.dt_scale


    def set_subordinator(self):
        '''
        Instantiates a multiplicative multifractal measure using lognormal multipliers,
        and loc and scale parameters, with support [0,T]. This is the model for trading time.  
        '''
        theta = Multifractal('lognormal', loc=self.loc, scale=self.scale, plot=False, support_endpoints=[0, self.T])
        return theta


    def set_subordinated(self):
        '''
        Sets the subordinated process for the different simulated models. If it's bm or fbm 
        only this function will run. 
        '''
        if self.sim_type == 'mmar_m' or 'bm':
            return BrownianMotion(drift=self.drift, scale=self.diffusion, t=self.T)
        elif self.sim_type == 'mmar' or 'fbm':
            return FractionalBrownianMotion(hurst=self.H, t=self.T)

    
    def get_increment(self, X, e):
        '''
        Returns the first difference of a vector. 
        '''
        diff = X[e:X.size:e] - X[:X.size-e:e]
        return diff


    def sim_bm(self, n):
        '''
        Simulates a discretized path with sample size n of a Brownian motion or Fractional brownian motion
        and returns both the time indexes and the realization of the process at that time. 
        '''
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
        to the __init__ function. By default it produces increments as defined by self.dt_scale
        '''
        #TODO check if this makes sense in the case of FBM. 
        #TODO drift for FBM
        #TODO README for the reason for the number of iterations. 
        self.subordinator.iterate(self.k)
        mu = self.subordinator.get_measure(self.dt_scale)
        mu_increment = np.sqrt(mu)

        mu_increment_size = mu_increment.size

        temp = self.sim_type

        if self.sim_type == 'mmar_m':
            self.sim_type = 'bm'
        elif self.sim_type == 'mmar':
            self.sim_type = 'fbm'

        s, times = self.sim_bm(mu_increment_size)
        s = self.get_increment(s, self.dt_scale)

        #Reset it because an instance of Multifractal keeps track of k. 
        self.subordinator = self.set_subordinator()
        self.sim_type = temp

        return (s*mu_increment, times)


    def sim_mmar_n(self, n, dt):
        res = np.array([])

        #Remembers the parameters of the instantiated object.
        cache = (self.T, self.dt_scale)

        #Sets k so it can produce at least the number of increments as given in n.
        self.k = math.ceil(math.log(n, 2))

        #Sets T so the max time is the same as the minimum required to produce n. 
        self.T = self.subordinator.b**self.k

        #Sets Δt so it produces realizations at the increments we want. 
        self.dt_scale = dt

        #Passes the parameters to the return process. 
        self.set_subordinated()
        
        y, _ = self.sim_mmar()
        res = np.append(res, y)

        #Resets the original parameters passed at object instantiation.  
        self.T = cache[0]
        self.k = math.ceil(math.log(self.T, 2))
        self.dt_scale = cache[1]
        self.set_subordinated()

        return res[:n]


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
        y = self.get_increment(y, self.dt_scale)
        plt.plot(x[1:], y)
        plt.xlabel('t') 
        plt.ylabel('X(t)')


    def plot_dist(self, n, increments=[1,7,30,180]):
        '''
        Plots the return distribution of a single realization of an mmar.
        '''
        fig, axes = plt.subplots(len(increments), 1, sharex='row', figsize=(20, 40))
        fig.subplots_adjust(hspace=0.5)

        for i in range(len(increments)):
            self.n = self.T // increments[i]

            if self.sim_type in ['mmar_m', 'mmar']:
                y, _ = self.sim_mmar()
                y = self.get_increment(y, increments[i])
            elif self.sim_type in ['bm', 'fbm']:
                y, _ = self.sim_bm(self.n)
                y = np.diff(y)
            bins = np.histogram(y, bins=math.ceil(np.sqrt(y.size)))  

            axes[i].hist(y, bins[1])
            axes[i].set_title(f"Return distribution at scale of {increments[i]} days.")
            axes[i].set_xlabel("X(t)")
            axes[i].set_ylabel("Density")