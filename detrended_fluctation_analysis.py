import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from repos.multifractal.multifractal import Multifractal


class FluctuationAnalysis():
    def __init__(self, data, nu):
        self.data = data
        self.diff_data = np.diff(data)
        self.nu = nu
        self.N = self.diff_data.size
        self.s = self.N // self.nu
        self.spl_data = self.split_data()


    def split_data(self):
        '''
        Splits a time series of differences into self.nu, equidistant ranges,
        and returns it as a self.nu x self.s matrix. 
        '''
        split_data = [self.diff_data[i:i+self.s] for i in range(0,self.N,self.s)]
        return np.array(split_data)
        

    def demean_data(self):
        '''
        Subtracts the mean of ranges with size self.s from the correpsonding ranges. 
        It returns self.nu x self.s matrix where in each row the data is equal to the original 
        data minus the mean of the data in that row. 
        '''
        means = np.mean(self.spl_data, axis=1)
        return [self.spl_data[i, :] - means[i] for i in range(means.size)]
    

    def integrate_series(self):
        '''
        Returns the profile (integrated series) of each of the demeaned ranges.
        '''
        return np.cumsum(self.demean_data(), axis=1)
    
    

    def rescaled_range(self):
        pass

