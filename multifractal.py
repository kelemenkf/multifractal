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
    def __init__(self, b, M, support_endpoints, E=1, k=0, mu=[1], P=[], r_type=""):
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
        self.P = np.array(P)
        self.r_type = r_type
    
    
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
    

    def get_measure(self):
        '''
        Return measure values for all box of size self.eps.
        '''
        return self.mu
    

    def update_P(self, M0, P, M):
        '''
        Update the array containing probabilities for the conservative construction.
        The relative probabilites are kept and rescaled so they always sum to 1.
        If say we have P = [0.5,0.3,0.2] and the random draw chooses 0.3, the next P will be 
        P = [0.71, 0, 0.28]. This keeps the ratio 0.5/0.2 = 0.71/0.28.
        '''
        p_index = np.where(M == M0)[0][0]
        P *= 1/(1-P[p_index])
        P[p_index] = 0
        assert np.isclose(np.sum(P),1)
        return P
        
        
    def draw_multiplier(self, M, P):
        '''
        Draws a single multiplier from a multinomial distribution with probabilities P. 
        '''
        return np.random.choice(M,size=1,p=P)[0]
    

    def conservative(self):
        '''
        Returns an array of multipliers where each one is drawn randomly according to P, 
        and keeps sum(M) == 1. This is the conservative construction. 
        '''
        P = np.copy(self.P)
        M = np.copy(self.M)
        result = []
        while len(result) < M.size:
            M0 = self.draw_multiplier(M, P)
            result.append(M0)
            if len(result) < M.size:
                P = self.update_P(M0,P,M)
        return result
    
    
    def canonical(self):
        '''
        Returns an array of multipliers where each is drawn randomly according to P,
        and keeps E[sum(M)] == 1. This is the canonical construction. 
        '''
        if not math.isclose(np.dot(self.M,self.P)*self.b, 1, rel_tol=1e-06):
            raise ValueError("E[ΣM] not equal to 1")
        M = np.random.choice(self.M,size=self.b,p=self.P)
        return M
    
    
    def multiply_measure(self, M, interval):
        '''
        Helper function for multiplying the measures with M. 
        '''
        temp = []
        for i in M:
            for j in interval:
                temp.append(i*j)
        return temp
    
    def multiply_measure_random(self, r_type):
        '''
        Multiplies the measure in each cell of interval with a random set of multipliers, 
        depending on r_type. 'Cons' is conservative, 'canon' is canonical. 
        '''
        temp = []
        for i in range(0,len(self.mu),2):
            interval = self.mu[i:i+2]
            if r_type == 'cons':
                M = self.conservative()
            elif r_type == 'canon':
                M = self.canonical()
            temp += self.multiply_measure(M, interval)
        return temp
        
    
    def iterate(self, k, plot=True):
        '''
        Does k iterations of the cascade, then Plots the resulting measure. 
        '''
        while k > 0:
            self.set_eps()
            self.set_support()
            if self.P.size > 0:
                temp = self.multiply_measure_random(self.r_type)
            else:
                temp = self.multiply_measure(self.M, self.mu)
            self.mu = np.array(temp)
            k -= 1
        if k <= 10 and plot == True:
            self.plot_density()
        
        
    def check_measure(self):
        '''
        Returns the sum of all measures on the set. Should sum to 1 if we want a probablity measure.
        '''
        return self.mu.sum()
    
    
    def plot_density(self):
        '''
        Plots the density function of the measure. 
        '''
        fig, ax = plt.subplots()
        plot = ax.bar(np.linspace(0,1,self.b**self.k,endpoint=False),self.mu/self.eps,1/(self.b**self.k),align='edge')
        plt.show()
            
    
    def convert_address(self,address):
        '''
        Converts a binary address, to an integer index.
        '''
        if self.k >= len(address):
            x = 0
            for i in range(1,len(address)+1):
                x += self.b**(-i) * int(address[i-1]) 
            start = np.nonzero(np.isclose(self.support,x))
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
                bar.set_height(self.mu[n]/self.eps)
                
            ax.set_ylim([0,max(self.mu/self.eps)+0.1*max(self.mu/self.eps)])
            self.iterate(1)

            return bars
        
        def init_func():
            return bars
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=500, init_func=init_func)
        ani.save(filename, writer="imagemagick")

        
    def coarse_alpha(self, address):
        '''
        Calculate the coarse Hölder exponent at a given interval.
        '''
        x = self.convert_address(address)
        alpha = math.log(self.mu[x]) / math.log(self.eps)
        return alpha
    
    
    def coarse_density(self, address):
        '''
        Calculate the coarse density at a given interval.
        '''
        x = self.convert_address(address)
        density = self.mu[x] / self.eps
        return density


    def baseb(self, n, b):
        '''
        Convert n in decimal to n in b.
        '''
        e = n//b
        q = n%b
        if n == 0:
            return '0'
        elif e == 0:
            return str(q)
        else:
            return self.baseb(e, b) + str(q)
    
    
    def get_alphas(self):
        '''
        Return an array of alpha exponents at the last stage of the iteration. 
        '''
        alphas = []
        for i in range(0,self.k):
            size = self.eps*self.b**i
            address = np.linspace(self.support_endpoints[0],self.support_endpoints[1],int(1/size),endpoint=False)
            N = len(address)
            alpha = []
            for i in range(N):
                x = self.baseb(i,self.b)
                if len(x) < math.log(N,self.b):
                    x = (int(math.log(N,self.b)) - len(x)) * '0' + x
                alpha.append(self.coarse_alpha(x))
            alphas.append(alpha)
        return alphas[0]
    
    
    def histogram_method_alpha_distribution(self):
        '''
        Plot the frequency distribution of alphas at the last stage of the iteration. 
        '''
        alphas = self.get_alphas()
        plt.xlabel("Coarse Hölder exponent (alpha)")
        plt.ylabel("Frequency")
        plt.hist(alphas,bins=self.k)
        
        
    def histogram_method_spectrum(self):
        '''
        Calculate the values of the multifractal spectrum of the measure. 
        '''
        alphas = self.get_alphas()
        
        bins = np.histogram(alphas,bins=self.k)
        
        N = bins[0]
        return (- np.log(N) / math.log(self.eps),bins[1])
    
    
    def histogram_method_spectrum_plot(self):
        '''
        Plot the multifractal spectrum of the measure. 
        '''
        y,x = self.histogram_method_spectrum()
        plt.plot(x[:-1],y)


    def cdf(self):
        return np.cumsum(self.mu)
    
    
    def plot_cdf(self):
        cdf = self.cdf()
        
        fig, ax = plt.subplots()
        plot = ax.bar(np.linspace(0,1,self.b**self.k,endpoint=False),cdf,1/self.b**self.k,align='edge')


    def partition_helper(self,q):
        '''
        Partition function for q-th moment, at the current level of coarse graining epsilon. 
        '''
        return np.sum(self.mu**q)
    

    def partition_function(self, k, q=5, plot=False):
        '''
        Calculate the partition function for an increasingly coarse-grained interval of size
        eps. q determines the maximum of moments (only integer), and k the number of iterations
        beyond the trivial first one. 
        '''
        data = np.ones((q,1))
        while k > 0:
            moments = []
            for q in range(1,q+1):
                chi = self.partition_helper(q)
                moments.append(chi)
            moments = np.array(moments)
            data = np.append(data, moments[:,np.newaxis], axis=1)
            self.iterate(1,plot=False)
            k -= 1
        return data
                
        
    def partition_plot(self, k, q):
        '''
        Plots the partition function for moments up until q (integers only) and for k iterations
        (trivial first one left out). 
        '''
        data = self.partition_function(k, q, plot=False)
        x = [self.eps * self.b**i for i in range(0,k)]
        print(x)
        for i in range(k):
            plt.plot(np.log(x), np.log(data[i,1:]), label=[f"{i+1} moment"])
        plt.legend()