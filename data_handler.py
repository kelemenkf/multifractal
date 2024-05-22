import math
import numpy as np
import datetime
import matplotlib.pyplot as plt 


class DataHandler():
    def __init__(self, data, delta_t=1.1, max_eps=183):
        self.data = data
        self.get_logprice()
        self.get_logreturns()
        self.delta_t = delta_t
        self.max_eps = max_eps
        self.drange = self.get_drange()
        self.eps = self.get_eps()
        self.X, self.X_abs = self.calc_x()


    def get_data(self):
        return self.X_abs, self.eps


    def get_drange(self):
        return max(self.data.index) - min(self.data.index)


    def get_df(self):
        return self.data
    

    def get_logprice(self, colname='Close'):
        self.data['logprice'] = self.data[colname].apply(lambda x: math.log(x))


    def get_logreturns(self, colname='logprice'):
        self.data['logreturn'] = self.data[colname].apply(lambda x: x - self.data.iloc[0][colname])


    def round_days(self, timedelta):
        total_seconds = timedelta.total_seconds()
        
        s = 86400
        
        days = math.ceil(total_seconds/s)
        
        return days
    

    def get_eps(self):
        drange = self.drange
        eps = [drange]

        while drange > datetime.timedelta(1):
            eps.append(drange/self.delta_t)
            drange /= self.delta_t

        for e in range(len(eps)):
            eps[e] = self.round_days(eps[e])

        eps = np.array(eps)
        eps = eps[eps < self.max_eps]
        eps = np.unique(eps)

        return eps


    def calc_x(self, colname='logreturn'):
        X = []
        X_abs = []
        for e in self.eps:
            row = []
            row_abs = []
            for i in range(e,len(self.data),e):
                row.append(self.data.iloc[i][colname] - self.data.iloc[i-e][colname])
                row_abs.append(abs(self.data.iloc[i][colname] - self.data.iloc[i-e][colname]))
            X.append(row)
            X_abs.append(row_abs)

        X = np.array(X, dtype=object)
        X_abs = np.array(X_abs, dtype=object)
    
        return X, X_abs
    

    def plot_x(self):
        plt.plot(self.data.index[1:], self.X[0],)
    
        
