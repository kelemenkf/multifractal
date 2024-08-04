import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import statsmodels.api as sm
import math

mlp.rcParams['figure.figsize'] = (20,10)

from repos.multifractal.src.time_series.time_series import TimeSeries as Nonstationary


class DFA(Nonstationary):
    def __init__(self, data, b=2, m=2, method='dfa', data_type='profile', nu_max=None):
        super().__init__(data, b, method, data_type=data_type, nu_max=nu_max)

        '''
        self.m - the degree of the polynomial fit. If m=2 the analysis is 
        a DFA2 analysis, which eliminates quadratic trends from the profile, 
        and linear trends from the series. 
        self.data_type - assumes that the time series is not integrated. 
        '''
        self.m = m
        #Checks condition that s >= m + 2
        assert np.all(self.s >= self.m + 2)
        self.alpha = self.calc_alpha()[1]


    def poly_fit_segment(self, segment):
        '''
        Fits a polynomial of degree m to the data found in the segment tuple.
        '''
        coeffs = np.polyfit(segment[0], segment[1], self.m)
        return coeffs
    

    def poly_coeffs_segment(self, Y, X):
        '''
        Returns a self.N_s[self.i] x (self.m + 1) matrix of polynomial coefficients for 
        each considered segment. 
        '''
        C = []
        for n in range(self.N_s[self.i]):
            segment = (X[n, :], Y[n, :])
            coeffs = self.poly_fit_segment(segment)
            C.append(coeffs)
        return np.array(C)
    

    def poly_vals_segment(self, Y, X):
        '''
        Returns a self.N_s[self.i] x self.s matrix of values of the fitted polynomial function,
        at each segment. 
        '''
        C = self.poly_coeffs_segment(Y, X)
        Y_fitted = []
        for n in range(self.N_s[self.i]):
            x = X[n, :]
            c = C[n]
            y = np.polyval(c, x)
            Y_fitted.append(y)
        return np.array(Y_fitted)


    def plot_poly(self, Y, X):
        '''
        Plots the fitted polynomial function of the data. 
        '''
        Y = self.poly_vals_segment(Y, X)
        Y = Y.flatten()
        X = X.flatten()
        plt.plot(X, Y)
        for i in range(0,X.size,self.s[self.i]):
            plt.axvline(X[i])


    def detrend_profile(self, Y, X):
        '''
        Detrends the profile of the data by subtracting the values of the fitted
        polynomial from the original data. 
        '''
        Y_fitted = self.poly_vals_segment(Y, X)
        return Y - Y_fitted
    

    def plot_Y_detrended(self, Y, X):
        '''
        Plots the detrended fluctuations of the time series.
        '''
        y = self.detrend_profile(Y, X)
        Y = y.flatten()
        X = X.flatten()
        plt.plot(X, Y)
        for i in range(X.size,0,-self.s[self.i]):
            plt.axvline(X[i-1])

    
    def plot_poly_with_data(self, Y, X, save=False, path="", name="", title=""):
        '''
        Plots the original data, the fitted polynomial functions and the detrended data
        in the same plot.  
        '''
        self.plot_poly(Y, X)
        plt.plot(X.flatten(), Y.flatten())
        plt.title(title)
        plt.xlabel("$t$")
        plt.ylabel("$X(t)$")
        self.plot_Y_detrended(Y, X)
        if save: 
            plt.savefig(path + '/' + name, dpi=300)
            plt.close()



    def squared_fluctuation(self, Y, X):
        '''
        Compuptes the mean squared fluctuation at each segment from the detrended data.
        This is equal to variance with self.s degrees of freedom. 
        '''
        Y_detrend = self.detrend_profile(Y, X)
        vars = np.var(Y_detrend, axis=1)
        return vars
    

    def mean_fluctuation(self):
        '''
        Calculates the mean fluctuation (the square root of the mean of squared fluctuations).
        '''
        fa_2 = np.concatenate((self.squared_fluctuation(self.spl_data, self.x_split),self.squared_fluctuation(self.spl_data_r, self.x_split_r)))
        mean = np.mean(fa_2)
        return np.sqrt(mean)


    def fluctuation_function(self):
        '''
        Returns the logarithm of FA_2(s) and self.s, for all scales defined by self.nu.
        '''
        fa = [self.mean_fluctuation()]
        for n in range(1, self.nu.size):
            self.i += 1
            self.reset_data()
            fa_s = self.mean_fluctuation()
            fa.append(fa_s) 
        self.i = 0
        self.reset_data()
        return np.log(fa), np.log(self.s)


    def calc_alpha(self):
        '''
        Returns alpha, that is the slope of the logarithm of FA_2(s) and self.s.
        '''
        y, x = self.fluctuation_function()
        x = sm.add_constant(x)
        model = sm.OLS(y, x)  
        result = model.fit()
        return result.params


    def plot_fa(self):
        '''
        Plots the logarithms of the FA_2(s) and self.s, and the best fit line. 
        '''
        y, x = self.fluctuation_function()
        plt.scatter(x, y)

        params = self.calc_alpha()
        best_fitted_y = params[0] + params[1] * x

        plt.plot(x, best_fitted_y, label=f"Slope of line: {params[1]}")
        plt.legend()


    #TODO correction function