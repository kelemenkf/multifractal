import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import statsmodels.api as sm

mlp.rcParams['figure.figsize'] = (16,9)

from repos.multifractal.stationary import Stationary as Nonstationary


class DFA(Nonstationary):
    def __init__(self, data, b=2, nu=5, m=2):
        super().__init__(data, b, nu)

        '''
        self.m - the degree of the polynomial fit. If m=2 the analysis is 
        a DFA2 analysis. 
        '''
        self.m = m
        self.alpha = self.calc_alpha()


    def poly_fit_segment(self, segment):
        '''
        Fits a polynomial of degree m to the data found in the segment tuple.
        '''
        coeffs = np.polyfit(segment[0], segment[1], self.m)
        return coeffs
    

    def poly_coeffs_segment(self, Y, X):
        '''
        Returns a self.N_s x self.m + 1 matrix of polynomial coefficients for 
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
        Returns a self.N_s x self.s matrix of values of the fitted polynomial function,
        at each segment. 
        '''
        C = self.poly_coeffs_segment(Y, X)
        Y_fitted = []
        for n in range(self.N_s[self.i]):
            x = self.x_split[n, :]
            c = C[n]
            y = np.polyval(c, x)
            Y_fitted.append(y)
        return np.array(Y_fitted)


    def plot_poly(self):
        '''
        Plots the fitted polynomial function and the scatter of the data together. 
        '''
        Y = self.poly_vals_segment(self.spl_data, self.x_split)
        Y = Y.flatten()
        X = self.x_split.flatten()
        # plt.scatter(X, self.spl_data.flatten())
        plt.plot(X, Y)
        for i in range(0,X.size-1,self.s[self.i]):
            plt.axvline(X[i])


    def detrend_profile(self, Y, X):
        '''
        Detrends the profile of the data by subtracting the values of the fitted
        polynomial from the original data. 
        '''
        Y_fitted = self.poly_vals_segment(Y, X)
        return Y - Y_fitted
    

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
        fa_2 = self.squared_fluctuation(self.spl_data, self.x_split) + self.squared_fluctuation(self.spl_data_r, self.x_split_r)
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
            fa.append(self.mean_fluctuation()) 
        self.i = 0
        self.reset_data()
        return np.log(fa), np.log(self.s)


    def plot_fa(self):
        '''
        Plots the logarithms of the FA_2(s) and self.s. 
        '''
        y, x = self.fluctuation_function()
        plt.plot(x, y)


    def calc_alpha(self):
        '''
        Returns alpha, that is the slope of the logarithm of FA_2(s) and self.s.
        '''
        y, x = self.fluctuation_function()
        x = sm.add_constant(x)
        model = sm.OLS(y, x)  
        result = model.fit()
        return result.params[1]
