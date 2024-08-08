import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from scipy.misc import derivative 
from scipy.optimize import curve_fit


from repos.multifractal.src.time_series.detrended_fluctuation_analysis import DFA

class MF_DFA(DFA):
    def __init__(self, data, b=2, method='mf_dfa', m=2, data_type='diff', q=[-5,5], gran=0.1, modified=False, nu_max=None):
        super().__init__(data, b=b, method=method, m=m, data_type=data_type, nu_max=nu_max) 

        '''
        self.data_type - assumes that the time series is not integrated, but the TimeSeries base class
        integrates it. 
        self.modified - If True it takes the sum of the profile. This will lead to better estimations of h(q) near
        h(q) = 0.
        '''
        self.q_range = np.linspace(q[0],q[-1],int((q[1]-q[0])/gran)+1)
        if modified:
            self.data = self.modified_data()
            self.reset_data()
        self.fa_q = self.fluctuation_functions()
        self.h_q = self.calc_h_q()
        self.spectrum = self.calc_spectrum(self.h_q)


    def modified_data(self):
        '''
        Returns the modified data for the modified MF-DFA which is more accurate 
        near h(q) = 0.
        '''
        data = np.cumsum(self.data - np.mean(self.data))
        return data


    def q_fluctuation_0(self):
        '''
        Calculates the fluctuation function with q = 0.
        '''
        f_2 = np.concatenate((self.squared_fluctuation(self.spl_data, self.x_split), self.squared_fluctuation(self.spl_data_r, self.x_split_r)))
        fa_0 = np.exp(np.sum(np.log(f_2)) / (self.N_s[self.i] * 4))
        return fa_0


    def q_fluctuation(self, q=[-5,5], gran=0.1):
        '''
        Obtain the qth order fluctuation function.
        '''
        fa = []
        fa_0 = self.q_fluctuation_0()
        for q in self.q_range:
            if q == 0:
                fa.append(fa_0)
            else:
                f_2 = np.concatenate((self.squared_fluctuation(self.spl_data, self.x_split), self.squared_fluctuation(self.spl_data_r, self.x_split_r)))
                fa_q = np.mean((f_2**(q/2)))**(1/q)
                fa.append(fa_q)
        return fa


    def fluctuation_functions(self):
        '''
        Calculates the fluctation function for different scales s, and for all orders q
        at each scale.
        '''
        fa_q = [self.q_fluctuation()]
        for n in range(1, self.nu.size):
            self.i += 1
            self.reset_data()
            fa_q.append(self.q_fluctuation())
        self.i = 0
        self.reset_data()
        return np.array(fa_q)
    

    def plot_f_q_s(self, q):
        '''
        Function to individually plot the data and the best fitted line on a 
        logarithmic scale of F_q(s) for a given q, specified by the parameter. 
        '''
        q_index = np.where(np.round(self.q_range,1) == q)
        print(q_index)
        data = self.fa_q[:, q_index]
        y = np.log(data)
        x = np.log(self.s)
        params = self.get_slope(y, x)

        plt.scatter(x, y)
        print(params)
        best_fit_y = params[0] + (params[1] * x)

        plt.plot(x, best_fit_y)
        plt.legend(f"{q}")


    def get_slope(self, y, x):
        '''
        Calculates slope of an ols regression of x on y. Constant is added. 
        '''
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        return results.params
    

    def calc_h_q(self):
        '''
        Calculates the h(q) function the generalized Hurst exponetns, describing the 
        scaling behaviour of the fluctuations. 
        '''
        h_q = {}
        data = self.fa_q
        for q in range(data.shape[1]):
            h = self.get_slope(np.log(data[:, q]), np.log(self.s))[1]
            h_q.update({self.q_range[q]:h})
        return h_q
    

    def plot_h_q(self, save=False, path="", name=""):
        '''
        Plots the h(q) function.
        '''
        plt.plot(list(self.h_q.keys()), list(self.h_q.values()))
        plt.xlabel("$q$")
        plt.ylabel("$h(q)$")
        if save:
            plt.savefig(path + '/' + name, dpi=300)
            plt.close()


    def plot_fa(self, save=False, path="", name="", title=""):
        ''' 
        Plots the fluctuation functions for the different qs. 
        '''
        fa_q = self.fluctuation_functions()
        for q in np.arange(0, 401, 10):
            plt.plot(np.log(self.s), np.log(fa_q[:,q]), label=[f"{q}"])
        plt.xlabel('$ln(s)$')
        plt.ylabel('$ln(F_q(s))$')
        plt.title(title)
        if save:
            plt.savefig(path + '/' + name, dpi=300)
            plt.close()


    def scaling_function(self):
        '''
        Plots the scaling function calculated from the estimated generalized Hurst 
        function.
        '''
        tau = np.array(list(self.h_q.keys())) * np.array(list(self.h_q.values())) - 1
        return tau
    

    def plot_scaling_function(self):
        tau = self.scaling_function()
        plt.plot(self.h_q.keys(), tau)
        plt.xlabel("$q$")
        plt.ylabel("τ(q)")

    
    def legendre(self):
        '''
        Legendre transform the tau(q) function and returns the spectrum f(alpha).
        '''
        alphas = self.alphas(self.h_q)
        tau = self.scaling_function()
        q = list(self.h_q.keys())
        f_alpha = {'f':np.array([]), 'alpha':np.array([])}
        for i in range(len(alphas)):
            f = alphas[i] * q[i] - tau[i]
            f_alpha['f'] = np.append(f_alpha['f'], f)
            f_alpha['alpha'] = np.append(f_alpha['alpha'], alphas[i])
        return f_alpha
    

    def fit_h_q(self, h_q):
        new_series = np.polynomial.polynomial.Polynomial.fit(list(h_q.keys()), list(h_q.values()), deg=4)
        return new_series.convert().coef
    

    def get_slopes(self, h_q):
        coeffs = self.fit_h_q(h_q)
        y = np.polyval(np.flip(coeffs), list(h_q.keys()))
        def f(x):
            return coeffs[-1]*x**4 + coeffs[-2]*x**3 + coeffs[-3]*x**2 + coeffs[-4]*x + coeffs[-5]
        slopes = []
        for i in list(h_q.keys()):
            slopes.append(derivative(f, i, dx=1e-6))
        return slopes


    def alphas(self, h_q):
        '''
        Calculates the alpha values needed to obtain the specturm f(α)
        '''
        slopes = self.get_slopes(h_q)
        return np.array(list(h_q.values())) + (np.array(list(h_q.keys())) * slopes)


    def calc_spectrum(self, h_q):
        '''
        Calculates the values taken by the multifractal spectrum. 
        '''
        alphas = self.alphas(h_q)
        f = (np.array(list(h_q.keys())) * (alphas - np.array(list(h_q.values())))) + 1
        return (f, alphas)
    

    def plot_spectrum(self, h_q, save=False, path="", name=""):
        '''
        Plots the multifractal spectrum. 
        '''
        f, alphas = self.calc_spectrum(h_q)
        plt.plot(alphas, f)
        plt.title('Multifractal specturm')
        plt.xlabel('$alpha$')
        plt.ylabel('$f(alpha)$')
        if save:
            plt.savefig(path + '/' + name, dpi=300)
            plt.close()


    def fit_spectrum(self, h_q):
        '''
        Fits a parabola to the estimated values of the spectrum. 
        '''
        f, alphas = self.calc_spectrum(h_q)
        new_series = np.polynomial.polynomial.Polynomial.fit(alphas, f, deg=2)
        return new_series


    def delta_alpha(self, h_q):
        '''
        Calculates delta alpha for h_q.
        '''
        new_series = self.fit_spectrum(h_q)
        roots = np.polynomial.polynomial.polyroots(new_series.convert().coef)
        return abs(roots[0] - roots[1])
    

    def f_P(self, alpha, alpha_0, H):
        '''
        Define the functinal form of the spectrum of the price process. 
        '''
        return 1 - ((alpha - alpha_0)**2 / (4 * H * (alpha_0 - H)))
    

    def fit_P_spectrum(self, h_q):
        '''
        Fits the spectrum function to the data obtained by the Legendre transform. 
        Returns alpha0, the maximum value of the parabola and the most likely exponent
        for a given time interval. 
        '''
        f, alphas = self.calc_spectrum(h_q)
        H = h_q[2.0]

        plt.scatter(alphas, f)

        params, _ = curve_fit(lambda alpha, alpha_0: self.f_P(alpha, alpha_0, H), alphas, f)

        return params[0]  
    

    def plot_fitted_f_alpha(self, h_q):
        f, alphas = self.calc_spectrum(h_q)
        alpha_0 = self.fit_P_spectrum(h_q)
        H = h_q[2.0]
        f_fitted = self.f_P(alphas, alpha_0, H)
        print(f_fitted)
        print(alphas)
        plt.plot(alphas, f_fitted)

    
    def get_lambda(self, h_q):
        '''
        Returns the estimated mean of the distribution of a lognormal multplier. 
        '''
        alpha_0 = self.fit_P_spectrum(h_q)
        H = h_q[2.0]
        return alpha_0  / H
    

    def get_sigma(self, h_q):
        '''
        Returns the estimated standard deviation of the distribution of a lognormal multiplier. 
        '''
        return np.sqrt(2 *  (self.get_lambda(h_q) - 1) /  math.log(self.b))