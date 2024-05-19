import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm

mpl.rcParams['figure.figsize'] = (20,10)

from .multifractal import Multifractal

class MethodOfMoments(Multifractal):
    def __init__(self, b, M, support_endpoints, q=5, gran=0.1, analytic=False, E=1, k=0, mu=[1], P=[], r_type=""):
        super().__init__(b, M, support_endpoints, E, k, mu, P, r_type)

        self.q = q
        self.gran = gran
        self.analytic = analytic
        self.data = self.partition_function()
        if not self.analytic:
            self.tau_q = self.calc_tau_q()
        else: 
            self.tau_q = self.calc_tau_q_binomial()
        self.f_alpha = self.legendre()


    def partition_helper(self, q):
        '''
        Partition function for q-th moment, at the current level of coarse graining epsilon. 
        '''
        return np.sum(self.mu**q)
       

    def partition_function(self, plot=False):
        '''
        Calculate the partition function for an increasingly coarse-grained interval of size
        eps. q determines the highest moment calculated (only integer), and k the number of iterations
        beyond the trivial first one. 
        '''
        q_range = np.linspace(1,self.q,int(self.q/self.gran))
        data = np.ones((q_range.size,1))
        k = self.k
        while k > 0:
            moments = []
            for q in q_range:
                chi = self.partition_helper(q)
                moments.append(chi)
            moments = np.array(moments)
            data = np.append(data, moments[:,np.newaxis], axis=1)
            self.iterate(1,plot=plot)
            k -= 1
        return (np.flip(data[:,1:], axis=1), q_range)
                
        
    def partition_plot(self):
        '''
        Plots the partition function for moments up until q (integers only) and for k iterations
        (trivial first one left out). 
        '''
        data = self.data[0]
        x = [self.eps * self.b**i for i in range(1,self.k-1)]
        print(data, x)
        for i in range(data.shape[0]):
            plt.scatter(np.log(x), np.log(data[i,:]), label=f"{i+1} moment")
        plt.xlabel("log(eps)")
        plt.ylabel("log(S)")
        plt.legend()


    def get_slope(self, y, x):
        '''
        Calculates slope of an ols regression of x on y. Constant is added. 
        '''
        x = sm.add_constant(x)
        model = sm.OLS(y,x)
        results = model.fit()
        return results.params[1]
        
        
    def calc_tau_q(self):
        '''
        Calculates the moment scaling function - tau(q) - of the measure in a range given by self.q, 
        with granularity given by self.gran.
        '''
        data, q_range = self.data
        tau_q = {}
        x = [self.eps * self.b**i for i in range(1, data.shape[1]+1)]
        for i in range(data.shape[0]):
            tau = self.get_slope(np.log(data[i,:]),np.log(x))
            tau_q.update({q_range[i]:tau})
        return tau_q
    
        
    def plot_tau_q(self):
        '''
        Plots the moment scaling function. 
        '''
        plt.plot(self.tau_q.keys(), self.tau_q.values())
        plt.xlabel('q')
        plt.ylabel('tau')


    def discrete_slopes(self):
        alphas = []
        tau = list(self.tau_q.values())
        q = list(self.tau_q.keys())
        for i in range(1,len(self.tau_q)):
            alpha = (tau[i] - tau[i-1]) / (q[i] - q[i-1])
            alphas.append(alpha)
        return alphas
    
    
    def legendre(self):
        alphas = self.discrete_slopes()
        tau = list(self.tau_q.values())
        q = list(self.tau_q.keys())
        f_alpha = {}
        for i in range(len(alphas)):
            f = alphas[i] * q[i] - tau[i]
            f_alpha.update({alphas[i]:f})
        return f_alpha
    
    
    def plot_f_alpha(self):
        plt.plot(list(self.f_alpha.keys()), list(self.f_alpha.values()))
        plt.xlabel('alpha')
        plt.ylabel('f(alpha)')

    
    def calc_tau_q_binomial(self):
        q_range = np.linspace(0,self.q,int(self.q/self.gran),endpoint=False)
        tau_q = {}
        for q in q_range:
            tau = -math.log(self.M[0]**q+self.M[1]**q,2)
            tau_q.update({q:tau})
        return tau_q

