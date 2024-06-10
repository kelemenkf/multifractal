import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from repos.multifractal.multifractal import Multifractal


class FluctuationAnalysis():
    def __init__(self, data, b=2, nu=np.array([1,2,3,4,5])):
        self.data = data
        self.diff_data = np.diff(data)
        self.b = b
        self.nu = nu
        self.N = self.diff_data.size
        self.s = self.N // (self.b**self.nu)
        self.i = 0
        self.spl_data = self.split_data()


    def split_data(self):
        '''
        Splits a time series of differences into self.nu, equidistant ranges,
        and returns it as a self.nu x self.s matrix. 
        '''
        split_data = [self.diff_data[i:i+self.s[self.i]] for i in range(0,self.N,self.s[self.i])]
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
    

    def get_ranges(self):
        '''
        Returns the maximum minus the minimum of the deviate series in each range.
        '''
        return np.max(self.integrate_series(), axis=1) - np.min(self.integrate_series(), axis=1)


    def get_std(self):
        '''
        Returns the standard deviation of the deviate series in each range. 
        '''
        return np.std(self.integrate_series(), axis=1)
    

    def rescaled_range(self):
        '''
        Returns the rescaled range for each range in the time series. 
        '''
        return self.get_ranges() / self.get_std()


    def average_rescaled_range(self):
        '''
        Returns the R/S value of a time series at a given scale self.s
        '''
        return np.mean(self.rescaled_range())
    

    def fluctuation_function(self):
        means = []
        for n in self.nu.size:
            means.append(self.average_rescaled_range()) 
            self.i = n           
        self.i = 0
        return means, self.s