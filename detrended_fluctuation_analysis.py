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
            segment = (self.x_split[n], self.spl_data[n])
            coeffs = self.poly_fit_segment(segment)
            C.append(coeffs)
        return np.array(C)
    

    def poly_calc_segment(self):
        C = self.poly_coeffs_segment()
        Y = np.empty((0,3))
        for n in range(self.N_s[self.i]):
            x = self.x_split[n, :]
            c = C[n]
            y = np.polyval(c, x)
            plt.plot(x, y)
            plt.scatter(self.x_split[n, :], self.spl_data[n, :])
            Y = np.append(Y, y)
        return Y


    def plot_poly(self, Y):
        plt.scatter()
        plt.plot()