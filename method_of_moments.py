import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit


mpl.rcParams['figure.figsize'] = (20,10)

from .multifractal import Multifractal

class MethodOfMoments(Multifractal):
    def __init__(self, b, M, support_endpoints, q=[-5,5], gran=0.1, analytic=False, X=np.array([]), delta_t=np.array([]), E=1, k=0, iter=0, mu=[1], P=[], r_type="", loc=0, scale=1):
        super().__init__(b, M, support_endpoints, E, k, mu, P, r_type, loc, scale)

        self.q = q
        self.gran = gran 
        self.q_range = np.linspace(self.q[0],self.q[-1],int((self.q[1]-self.q[0])/self.gran)+1)
        self.iter = iter
        self.analytic = analytic
        self.X = X
        self.delta_t = delta_t
        self.data = self.partition_function()
        if not self.analytic:
            self.tau_q = self.calc_tau_q()
        else: 
            self.tau_q = self.calc_tau_q_multinomial()
        self.f_alpha = self.legendre()
        self.H = self.get_H()
        self.alpha_0 = self.fit_spectrum()


    def partition_helper(self, X):
        '''
        Partition function for q-th moment, at the current level of coarse graining epsilon. 
        '''
        moments = []
        for q in self.q_range:
            chi = np.sum(X**q)
            moments.append(chi)
        moments = np.array(moments)
        return moments
       

    def partition_function(self, plot=False):
        '''
        Calculate the partition function for an increasingly coarse-grained interval of size
        eps. q determines the highest moment calculated (only integer), and k the number of iterations
        beyond the trivial first one. 
        '''
        data = np.ones((self.q_range.size,1))
        if self.X.size > 0:
            for t in range(len(self.delta_t)):
                moments = self.partition_helper(self.X[t])
                data = np.append(data, moments[:,np.newaxis], axis=1)
            #First column is excluded because in this context it corresponds to nothing. 
            return data[:,1:]
        else:
            k = self.iter
            while k > 0:
                moments = self.partition_helper(self.mu)
                data = np.append(data, moments[:,np.newaxis], axis=1)
                self.iterate(1,plot=plot)
                k -= 1
            return np.flip(data[:,1:], axis=1)
                
        
    def partition_plot(self, renorm=False):
        '''
        Plots the partition function for moments up until q (integers only) and for k iterations
        (trivial first one left out). 
        '''
        data = self.data
        if self.delta_t.size == 0:
            x = [self.eps * self.b**i for i in range(1,self.k+1)]
        else:
            x = self.delta_t
        if renorm:
            offsets = np.log(self.data[:,0])
        else:
            offsets = np.zeros(len(self.q_range))
        for i in range(len(self.q_range)):
            plt.plot(np.log(x), np.log(data[i,:]) - offsets[i], label=f"{self.q_range[i]} moment")
        plt.xlabel("log(eps)")
        plt.ylabel("log(S)")
        plt.legend()


    def get_slope(self, y, x):
        '''
        Calculates slope of an ols regression of x on y. Constant is added. 
        '''
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        return results.params[1]
        
        
    def calc_tau_q(self):
        '''
        Calculates the moment scaling function - tau(q) - of the measure in a range given by self.q, 
        with granularity given by self.gran.
        '''
        tau_q = {}
        if self.delta_t.size == 0:
            x = [self.eps * self.b**i for i in range(1,self.k+1)]
        else:
            x = self.delta_t
        for i in range(self.data.shape[0]):
            tau = self.get_slope(np.log(self.data[i,:]),np.log(x))
            tau_q.update({self.q_range[i]:tau})
        return tau_q
    
        
    def plot_tau_q(self):
        '''
        Plots the moment scaling function. 
        '''
        plt.plot(self.tau_q.keys(), self.tau_q.values())
        plt.xlabel('q')
        plt.ylabel('tau')


    def discrete_slopes(self):
        '''
        Calculates a discrete apporximation of the derivative of the tau(q) function. 
        '''
        alphas = []
        tau = list(self.tau_q.values())
        q = list(self.tau_q.keys())
        for i in range(1,len(self.tau_q)):
            alpha = (tau[i] - tau[i-1]) / (q[i] - q[i-1])
            alphas.append(alpha)
        return alphas
    
    
    def legendre(self):
        '''
        Legendre transform the tau(q) function and returns the spectrum f(alpha).
        '''
        #TODO check weird f values around 1.
        alphas = self.discrete_slopes()
        tau = list(self.tau_q.values())
        q = list(self.tau_q.keys())
        f_alpha = {'f':np.array([]), 'alpha':np.array([])}
        for i in range(len(alphas)):
            f = alphas[i] * q[i] - tau[i]
            f_alpha['f'] = np.append(f_alpha['f'], f)
            f_alpha['alpha'] = np.append(f_alpha['alpha'], alphas[i])
        return f_alpha
    
    
    def plot_f_alpha(self):
        '''
        Plots the estimated multifractal spectrum. 
        '''
        plt.scatter(self.f_alpha['alpha'], self.f_alpha['f'])
        plt.xlabel('alpha')
        plt.ylabel('f(alpha)')

    
    def calc_tau_q_multinomial(self):
        '''
        Calculates the tau(q) of a b-nomial meausre which has an exact formula. 
        '''
        tau_q = {}
        for q in self.q_range:
            sigma = 0
            for m in self.M:
                sigma += m**q
            tau = -math.log(sigma, self.b)
            tau_q.update({q:tau})
        return tau_q


    def get_H(self):
        '''
        Returns the H index that is the value where the scaling function is the closest to 0.
        '''
        #TODO better approximation for H.
        L = list(self.tau_q.values())
        i = L.index(min(L, key=lambda x: abs(x - 0)))
        return 1 / list(self.tau_q.keys())[i]
    

    def get_lambda(self):
        '''
        Returns the estimated mean of the distribution of a lognormal multplier. 
        '''
        return self.alpha_0 / self.H
    

    def get_sigma(self):
        '''
        Returns the estimated standard deviation of the distribution of a lognormal multiplier. 
        '''
        return np.sqrt(2 *  (self.get_lambda() - 1) /  math.log(self.b))


    def f_P(self, alpha, alpha_0, H):
        '''
        Define the functinal form of the spectrum of the price process. 
        '''
        return 1 - ((alpha - alpha_0)**2 / (4 * H * (alpha_0 - H)))


    def fit_spectrum(self):
        '''
        Fits the spectrum function to the data obtained by the Legendre transform. 
        Returns alpha0, the maximum value of the parabola and the most likely exponent
        for a given time interval. 
        '''
        alpha_data = self.f_alpha['alpha']
        f_alpha_data = self.f_alpha['f']
        H = self.H

        params, covariance = curve_fit(lambda alpha, alpha_0: self.f_P(alpha, alpha_0, H), alpha_data, f_alpha_data)

        return params[0]  
    

