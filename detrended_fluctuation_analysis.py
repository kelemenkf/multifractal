import numpy as np

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
        C = np.array([])
        for n in range(self.N_s[self.i]):
            segment = (self.x_split[n], self.spl_data[n])
            print(segment)
            coeffs = self.poly_fit_segment(segment)
            C = np.append(C, coeffs)
        return C