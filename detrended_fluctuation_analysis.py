import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp

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
        self.C = self.poly_coeffs_segment()
        self.Y = self.poly_vals_segment()
        self.Y_detrend = self.detrend_profile()


    def poly_fit_segment(self, segment):
        '''
        Fits a polynomial of degree m to the data found in the segment tuple.
        '''
        coeffs = np.polyfit(segment[0], segment[1], self.m)
        return coeffs
    

    def poly_coeffs_segment(self):
        '''
        Returns a self.N_s x self.m + 1 matrix of polynomial coefficients for 
        each considered segment. 
        '''
        C = []
        for n in range(self.N_s[self.i]):
            segment = (self.x_split[n, :], self.spl_data[n, :])
            coeffs = self.poly_fit_segment(segment)
            C.append(coeffs)
        return np.array(C)
    

    def poly_vals_segment(self):
        '''
        Returns a self.N_s x self.s matrix of values of the fitted polynomial function,
        at each segment. 
        '''
        C = self.poly_coeffs_segment()
        Y = []
        for n in range(self.N_s[self.i]):
            x = self.x_split[n, :]
            c = C[n]
            y = np.polyval(c, x)
            Y.append(y)
        return np.array(Y)


    def plot_poly(self):
        '''
        Plots the fitted polynomial function and the scatter of the data together. 
        '''
        Y = self.Y.flatten()
        X = self.x_split.flatten()
        plt.scatter(X, self.spl_data.flatten())
        plt.plot(X, Y)
        # plt.axvline([self.x[::self.s[self.i]]])


    def detrend_profile(self):
        '''
        Detrends the profile of the data by subtracting the values of the fitted
        polynomial from the original data. 
        '''
        return self.spl_data - self.Y
    

    def squared_fluctuation(self):
        '''
        Compuptes the mean squared fluctuation at each segment from the detrended data.
        This is equal to variance with self.s degrees of freedom. 
        '''
        vars = np.var(self.Y_detrend, axis=1)
        return vars
    

    def mean_fluctuation(self):
        '''
        Calculates the mean fluctuation (the square root of the mean of squared fluctuations).
        '''
        fa_2 = self.squared_fluctuation(self.spl_data) + self.squared_fluctuation(self.spl_data_r)
        mean = np.mean(fa_2)
        return np.sqrt(mean)


    def fluctuation_function(self):
        fa = []