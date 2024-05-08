import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import sys


class Multifractal():
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
        self.k += 1
        self.eps = self.b**(-self.k)
       
    
    def set_support(self):
        self.support = np.linspace(self.support_endpoints[0],self.support_endpoints[1],self.b**self.k,endpoint=False)
    
    
    def iterate(self, k):
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
        return self.mu.sum()
    
    
    def plot_measure(self):
        fig, ax = plt.subplots()
        plot = ax.bar(np.linspace(0,1-(1/self.b**self.k),self.b**self.k),self.mu,1/(self.b**self.k),align='edge')
        plt.show()
            
    
    def convert_address(self,address):
        if self.k >= len(address):
            x = 0
            for i in range(1,len(address)+1):
                x += self.b**(-i) * int(address[i-1]) 
            start = np.where(self.support == x)[0][0]
            print(start)
            return start
        else: 
            raise ValueError("Not enough iterations")
    
    
    def plot_interval(self,address):
        fig,ax = plt.subplots()
        start = self.convert_address(address)
        h = len(address)
        if (start + 1) == len(self.support):
            end = self.support_endpoints[1]
        else:
            end = self.support[start+1]
        print(end,self.b**self.k,self.mu[start:])
        plot = ax.bar(np.linspace(self.support[start],end,self.b**(self.k-h)),self.mu[start:],1/(self.b**self.k),align='edge')
        plt.show()
        
        
    def animate(self, frames):
        fig, ax = plt.subplots()
        
        bars = ax.bar(0,1,1,align='edge')
        
        def animate_frame(frame):
            ax.clear()
            
            self.iterate(1)

            while self.k > len(bars):
                bars.append(ax.bar(0,1,1,align='edge'))
                
            for i, bar in enumerate(bars):
                bar.set_height(self.mu[i])
                bar.set_width(1/(self.b**self.k))
                
            return bars
        
        ani = animation.FuncAnimation(fig=fig, func=animate_frame, frames=frames, interval=120)
        plt.show()
