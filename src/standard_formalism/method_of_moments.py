import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
import sys

mpl.rcParams['figure.figsize'] = (15,5)

from repos.multifractal.src.multifractal import Multifractal


class MethodOfMoments(Multifractal):
    def __init__(self, M_type, M=[0.6,0.4], b=2, support_endpoints=[0,1], q=[-5,5], gran=0.1, analytic=False, X=np.array([]), delta_t=np.array([]), E=1, k=0, iter=0, mu=[1], P=[], r_type="", loc=0, scale=1, drift=None, diffusion=None):
        super().__init__(M, M_type, b, support_endpoints, E, k, mu, P, r_type, loc, scale)

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
        self.alpha_0 = self.fit_P_spectrum()
        self.theta_lambda = self.fit_theta_spectrum()
        if drift != None and drift in self.delta_t:
            self.drift = self.get_drift(drift)
        # else:
        #     raise ValueError('No data for that frequency')
        if diffusion != None and diffusion in self.delta_t:
            self.diffusion = self.get_diffusion(diffusion)
        # else:
        #     raise ValueError('No data for that frequency')


    def get_drift(self, n):
        delta_t_index = np.where(self.delta_t == n)[0][0]
        return np.mean(self.X[delta_t_index])
    

    def get_diffusion(self, n):
        delta_t_index = np.where(self.delta_t == n)[0][0]
        return np.std(self.X[delta_t_index])


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
        eps. q determines the highest moment calculated, and k the number of iterations
        beyond the trivial first one. 
        '''
        data = np.ones((self.q_range.size,1))
        X = np.abs(self.X)
        if self.X.size > 0:
            for t in range(len(self.delta_t)):
                moments = self.partition_helper(X[t])
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
                
        
    def partition_plot(self, renorm=False, save=False, path="", name=""):
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
            if self.q_range[i] in np.arange(0, 5.5, 0.5):
                plt.plot(np.log(x), np.log(data[i,:]) - offsets[i], label=f"{self.q_range[i]} moment")
        print(name)
        plt.xlabel("$ln(ε)$")
        plt.ylabel("$ln(S)$")
        plt.title("Partition function")
        plt.legend()
        if save:
            plt.savefig(path + "/" + name)
            plt.close()


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
    
        
    def plot_tau_q(self, save=False, path="", name=""):
        '''
        Plots the moment scaling function. 
        '''
        plt.plot(self.tau_q.keys(), self.tau_q.values())
        plt.title("Scaling function")
        plt.xlabel('$q$')
        plt.ylabel('$τ$')
        if save:
            plt.savefig(path + "/" + name)
            plt.close()


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
        #TODO check weird f values around 1. This may be due to inversion. 
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
        L = list(self.tau_q.values())
        i = L.index(min(L, key=lambda x: abs(x - 0)))
        return 1 / list(self.tau_q.keys())[i]
    

    #TODO better approximation for H.
    def fit_tau_q(self):
        q_data = list(self.tau_q.values())
        tau_data = list(self.tau_q.keys())
    

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


    #TODO check if this is the correct spectrum. 

    def f_P(self, alpha, alpha_0, H):
        '''
        Define the functinal form of the spectrum of the price process. 
        '''
        return 1 - ((alpha - alpha_0)**2 / (4 * H * (alpha_0 - H)))
    

    def f_theta(self, alpha, mean):
        '''
        Defines the functional form of the spectrum of trading time.
        '''
        return 1 - ((alpha - mean)**2 / (4 * (mean - 1)))


    def fit_P_spectrum(self):
        '''
        Fits the spectrum function to the data obtained by the Legendre transform. 
        Returns alpha0, the maximum value of the parabola and the most likely exponent
        for a given time interval. 
        '''
        alpha_data = self.f_alpha['alpha'][::-1]
        f_alpha_data = self.f_alpha['f'][::-1]
        H = self.H

        params, _ = curve_fit(lambda alpha, alpha_0: self.f_P(alpha, alpha_0, H), alpha_data, f_alpha_data)

        return params[0]  
    

    def fit_theta_spectrum(self):
        '''
        Fits the spectrum of the trading time obtained by the Legendre transform. 
        Should return lambda. 
        #TODO why doesn'it ? 
        '''
        alpha_data = self.f_alpha['alpha'][::-1]
        f_alpha_data = self.f_alpha['f'][::-1]

        params, _ = curve_fit(lambda alpha, mean: self.f_theta(alpha, mean), alpha_data, f_alpha_data)

        return params[0]
    

    def plot_fitted_f_alpha(self, save=False, path="", name=""):
        '''
        Creates plots displaying the estimated spectrum and the fitted parabola of the compound process, 
        and the fitted parabola of the subordinator (trading time). 
        '''
        fig, axes = plt.subplots(1, 2)

        alpha_data = self.f_alpha['alpha']
        alpha_theta_data = alpha_data / self.H

        f_P_alpha_data = self.f_P(alpha_data, alpha_0=self.alpha_0, H=self.H)
        f_theta_alpha_data = self.f_theta(alpha_theta_data, mean=self.get_lambda())

        axes[1].scatter(self.f_alpha['alpha'], self.f_alpha['f'])
        axes[1].plot(alpha_data, f_P_alpha_data)
        axes[1].set_xlabel("$α$")
        axes[1].set_ylabel("$f$")
        axes[1].set_title('Estimated values of the specturm from the Legendre transfrom and the fitted parabola')
        axes[0].plot(alpha_theta_data, f_theta_alpha_data)
        axes[0].set_title('Estimated spectrum of trading time')
        axes[0].set_xlabel("$α$")
        axes[0].set_ylabel("$f$")

        if save:
            fig.savefig(path + "/" + name)
            plt.close(fig)

