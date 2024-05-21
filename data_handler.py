import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import statsmodels.api as sm
from matplotlib.patches import Rectangle
import datetime
import sys


from .multifractal import Multifractal


class DataHandler():
    def __init__(self, data, delta_t=1.1, max_eps=183):
        self.data = data
        self.delta_t = delta_t
        self.max_eps = max_eps
        self.drange = self.get_drange()
        self.eps = self.get_eps()


    def get_drange(self):
        return max(self.data.index) - min(self.data.index)
    

    def round_days(timedelta):
        total_seconds = timedelta.total_seconds()
        
        s = 86400
        
        days = math.ceil(total_seconds/s)
        
        return days
    

    def get_eps(self):
        drange = self.drange
        eps = [drange]

        while drange > datetime.timedelta(1):
            eps.append(drange/1.1)
            drange /= 1.1

        for e in range(len(eps)):
            eps[e] = self.round_days(eps[e])

        eps = np.array(eps)
        eps = eps[eps < self.max_eps]
        eps = np.unique(eps)

        return eps


    def calc_x(self, data, colname='logprice'):
        X = []
        for e in self.eps:
            row = []
            for i in range(e,len(data),e):
                row.append(abs(data.iloc[i]['logprice'] - data.iloc[i-e]['logprice']))
            X.append(row)

        X = np.array(X, dtype=object)
        X = np.flip(X)
    
        return X
        
