import math
import numpy as np
import datetime


class DataHandler():
    def __init__(self, data, delta_t=1.1, max_eps=183):
        self.data = data
        self.get_logprice()
        self.delta_t = delta_t
        self.max_eps = max_eps
        self.drange = self.get_drange()
        self.eps = self.get_eps()
        self.X = self.calc_x()


    def get_data(self):
        return self.X, self.eps


    def get_drange(self):
        return max(self.data.index) - min(self.data.index)
    

    def get_logprice(self, colname='Close'):
        self.data['logprice'] = self.data[colname].apply(lambda x: math.log(x))


    def round_days(self, timedelta):
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


    def calc_x(self, colname='logprice'):
        X = []
        for e in self.eps:
            row = []
            for i in range(e,len(self.data),e):
                row.append(abs(self.data.iloc[i][colname] - self.data.iloc[i-e][colname]))
            X.append(row)

        X = np.array(X, dtype=object)
        X = np.flip(X)
    
        return X
        
