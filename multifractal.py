import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import sys


class Multifractal():
    '''
    Base class for a multifractal measure created using a multiplicative cascasde. 
    The cascade is b-nomial (e.g. binomial,trinomial etc.) with the b multipliers defined in M. 
    The set supporting the measure is Euclidean with the dimension E. 
    Plotting is only supported with E=1. The values of the measure is stored in mu, 
    and k denotes the number of iterations for the cascade for a given measure. 
    '''
    def __init__(self, b, M, support_endpoints, E, k=0, mu=[1]):
        self.b = b
        if sum(M) == 1:
            self.M = np.array(M, dtype='f')
        else:
            raise ValueError("Multipliers do not add to 1")
        self.E = E
        self.k = k
        self.mu = np.array(mu)
        self.eps = self.b**(-self.k)
        self.support_endpoints = support_endpoints
        self.support = np.linspace(support_endpoints[0],support_endpoints[1],self.b**self.k,endpoint=False)
    
    
    def __str__(self):
        return f"{self.mu}"
        
        
    def set_eps(self):
        '''
        Updates the number of iterations of a given instance, and updates scaling variable. 
        '''
        self.k += 1
        self.eps = self.b**(-self.k)
       
    
    def set_support(self):
        '''
        Updates the x-ticks of the support set.
        '''
        self.support = np.linspace(self.support_endpoints[0],self.support_endpoints[1],self.b**self.k,endpoint=False)
    
    
    def iterate(self, k):
        '''
        Does k iterations of the cascade, then Plots the resulting measure. 
        '''
        while k > 0:
            self.set_eps()
            self.set_support()
            temp = []
            for i in self.M:
                for j in self.mu:
                    temp.append(i*j)
            self.mu = np.array(temp)
            k -= 1
        self.plot_measure()
        
        
    def check_measure(self):
        '''
        Returns the sum of all measures on the set. Should sum to 1 if we want a probablity measure.
        '''
        return self.mu.sum()
    
    
    def plot_measure(self):
        '''
        Plots the multifractal on the whole set supporting it
        '''
        fig, ax = plt.subplots()
        plot = ax.bar(np.linspace(0,1-(1/self.b**self.k),self.b**self.k),self.mu,1/(self.b**self.k),align='edge')
        plt.show()
            
    
    def convert_address(self,address):
        '''
        Converts a binary address, to an integer index.
        '''
        if self.k >= len(address):
            x = 0
            for i in range(1,len(address)+1):
                x += self.b**(-i) * int(address[i-1]) 
            start = np.where(self.support == x)[0][0]
            return start
        else: 
            raise ValueError("Not enough iterations")
    
    
    def plot_interval(self,address):
        '''
        Plots the measure on a given inerval of the support. 
        '''
        fig,ax = plt.subplots()
        start = self.convert_address(address)
        h = len(address)
        if start + self.b**(self.k-h) == len(self.support):
            end = self.support_endpoints[1]
        else:
            end = self.support[start+self.b**(self.k-h)]
        plot = ax.bar(np.linspace(self.support[start],end,self.b**(self.k-h),endpoint=False),self.mu[start:start+self.b**(self.k-h)],1/(self.b**self.k),align='edge')
        plt.show()
        

    def animate(self, frames, filename):
        fig, ax = plt.subplots()
            
        bars = ax.bar(np.linspace(0,1,self.b**frames,endpoint=False),np.ones(self.b**frames),1/(self.b**frames),align='edge')
        
        def update(frame):
            for i, bar in enumerate(bars):
                n = int(i // (self.b**frames / self.b**self.k))
                bar.set_height(self.mu[n])
                
            ax.set_ylim([0,max(self.mu)+0.1*max(self.mu)])
            self.iterate(1)

            return bars
        
        def init_func():
            return bars
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=500, init_func=init_func)
        ani.save(filename, writer="imagemagick")

        
    def coarse_alpha(self, address):
        x = self.convert_address(address)
        alpha = math.log(self.mu[x]) / math.log(self.eps)
        return alpha
    
    
    def coarse_density(self, address):
        x = self.convert_address(address)
        density = self.mu[x] / self.eps
        return density