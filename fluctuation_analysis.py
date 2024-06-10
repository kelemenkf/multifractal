import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm

from repos.multifractal.stationary import Stationary


class FluctuationAnalysis(Stationary):
    def __init__(self, data, b=2, nu_max=5):
        super().__init__(data, b)

        self.nu_min = math.ceil(math.log(10,self.b))
        if self.nu_min > nu_max:
            raise ValueError("self.nu has to be larger")
        self.nu_max = nu_max
        self.nu = np.array(range(self.nu_min,self.nu_max))
        self.s = self.N // (self.b**self.nu)
        self.alpha = self.calc_alpha()


    def squared_fluctuation(self, data):
        fa_2 = []
        for range in data:
            fa_2.append((range[0] - range[-1])**2)
        return fa_2


    def mean_fluctuation(self):
        fa_2 = self.squared_fluctuation(self.spl_data) + self.squared_fluctuation(self.spl_data_r)
        print(len(fa_2))
        mean = np.mean(fa_2)
        return np.sqrt(mean)
    

    def fluctuation_function(self):
        fa = [self.mean_fluctuation()]
        for n in range(1, self.nu.size):
            self.i += 1
            self.spl_data = self.split_data(self.diff_data) 
            self.spl_data_r = self.split_data(np.flip(self.diff_data))
            fa.append( self.mean_fluctuation()) 
        return np.log(fa), np.log(self.s)


    def calc_alpha(self):
        y, x = self.fluctuation_function()
        x = sm.add_constant(x)
        model = sm.OLS(y, x)  
        result = model.fit()
        return result.params[1]  

    