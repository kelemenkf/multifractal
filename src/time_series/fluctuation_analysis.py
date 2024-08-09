import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm

from repos.multifractal.src.time_series.time_series import TimeSeries


class FluctuationAnalysis(TimeSeries):
    def __init__(self, data, b=2, method='fa', nu_max=8):
        super().__init__(data, b=b, method=method, nu_max=nu_max)

        self.alpha = self.calc_alpha()


    def squared_fluctuation(self, data):
        '''
        Returns the squared fluctuations of all ranges at a given scale.
        '''
        fa_2 = []
        for range in data:
            fa_2.append((range[0] - range[-1])**2)
        return fa_2


    def mean_fluctuation(self):
        '''
        Calculates the mean fluctuation (the square root of the mean of squared fluctuations).
        '''
        fa_2 = self.squared_fluctuation(self.spl_data) + self.squared_fluctuation(self.spl_data_r)
        mean = np.mean(fa_2)
        return np.sqrt(mean)
    

    def fluctuation_function(self):
        '''
        Returns the logarithm of FA_2(s) and self.s, for all scales defined by self.nu.
        '''
        fa = [self.mean_fluctuation()]
        for n in range(1, self.nu.size):
            self.scale_iterator += 1
            self.reset_data()
            fa.append(self.mean_fluctuation()) 
        self.scale_iterator = 0
        self.reset_data()
        return np.log(fa), np.log(self.scale_lengths)


    def plot_fa(self):
        '''
        Plots the logarithms of the FA_2(s) and self.s. 
        '''
        y, x = self.fluctuation_function()
        plt.plot(x, y)


    def calc_alpha(self):
        '''
        Calculates alpha, that is the slope of the logarithm of FA_2(s) and self.s
        '''
        y, x = self.fluctuation_function()
        x = sm.add_constant(x)
        model = sm.OLS(y, x)  
        result = model.fit()
        return result.params[1]


    