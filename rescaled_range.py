import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


from repos.multifractal.stationary import Stationary


class RescaledRange(Stationary):
    def __init__(self, data, b=2, nu_max=8, method='rs') -> None:
        super().__init__(data, b=b, nu_max=nu_max, method=method)

        self.spl_data = self.split_data(self.diff_data)
        self.H = self.calc_H()
        

    def demean_data(self):
        '''
        Subtracts the mean of ranges with size self.s from the correpsonding ranges. 
        It returns seflfb.**self.nu x self.s matrix where in each row the data is equal to the original 
        data minus the mean of the data in that row. 
        '''
        means = np.mean(self.spl_data, axis=1)
        return [self.spl_data[i, :] - means[i] for i in range(means.size)]
    

    def integrate_series(self):
        '''
        Returns the profile (integrated series) of each of the demeaned ranges.
        '''
        integrated_series = np.cumsum(self.demean_data(), axis=1)
        return integrated_series
    

    def get_ranges(self):
        '''
        Returns the maximum minus the minimum of the deviate series in each range.
        '''
        range = np.max(self.integrate_series(), axis=1) - np.min(self.integrate_series(), axis=1)
        return range


    def get_std(self):
        '''
        Returns the standard deviation of the deviate series in each range. 
        '''
        std = np.std(self.spl_data, axis=1)
        return std
    

    def rescaled_range(self):
        '''
        Returns the rescaled range for each range in the time series. 
        '''
        r_s = self.get_ranges() / self.get_std()
        return r_s


    def average_rescaled_range(self):
        '''
        Returns the average R/S value of a time series at a given scale self.s
        '''
        return np.mean(self.rescaled_range())
    

    def fluctuation_function(self):
        '''
        Calculates the fluctuation function for the available scales and returns, 
        the logarithm of the values. 
        '''
        means = [self.average_rescaled_range()]
        for n in range(1, self.nu.size):
            self.i += 1
            self.spl_data = self.split_data(self.diff_data) 
            means.append(self.average_rescaled_range()) 
        self.i = 0
        self.spl_data = self.split_data(self.diff_data)
        return np.log(means), np.log(self.s)
    

    def plot_fa_function(self):
        '''
        Plots the logarithm of the  calculated fluctuation function.
        '''
        rs_means, x = self.fluctuation_function()
        plt.plot(x, rs_means)
        plt.xlabel("log(x)")
        plt.ylabel("log(R/S)")


    def calc_H(self):
        '''
        Calculates the slope of the logarithm of the fluctuation function (H).
        '''
        rs_means, x = self.fluctuation_function()
        x = sm.add_constant(x)
        model = sm.OLS(rs_means, x)
        results = model.fit()
        return results.params[1]