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
        self.X = self.calc_x()


    def get_data(self):
        '''
        Return calculated increments with different delta_ts, and the logarithm of delta_t. 
        '''
        return self.X, self.eps


    def get_drange(self):
        '''
        Returns the range of the date of the dataset in timedelta format.
        '''
        return max(self.data.index) - min(self.data.index)


    def get_df(self):
        '''
        Returns the input data in dataframe format. 
        '''
        return self.data
    

    def get_logprice(self, colname='Close'):
        '''
        Calculates the logprice of a price series. Colname dentoes the name of the
        price column with 'Close' being a default from Yahoo finance. 
        '''
        self.data['logprice'] = self.data[colname].apply(lambda x: math.log(x))


    def get_logreturns(self, colname='logprice'):
        '''
        Calculates X(t), i.e., the series of cumulative returns since the first date in 
        the dataset. 
        '''
        self.data['logreturn'] = self.data[colname].apply(lambda x: x - self.data.iloc[0][colname])


    def round_days(self, timedelta):
        '''
        Rounds a timedelta value to the nearest day. 
        '''
        total_seconds = timedelta.total_seconds()
        
        s = 86400
        
        days = math.ceil(total_seconds/s)
        
        return days
    

    def get_eps(self):
        '''
        Returns an array of delta_t values where each element is determined by 
        a multiplicative factor so if it is 2 and there are 2000 days in the dataset, 
        it will return 2000, 1000, 500, 250 etc. These will be the values for the 
        different increments in the partition function. Self.max_eps determines the largest
        value of the increment which defaults to 183, i.e., half a year. 
        '''
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
        '''
        Calculates the increments of the series X(t), with different values of
        delta_t as determined by self.eps. It does this both in absolute value for the 
        partition function, and just with a regular difference for the plot. 
        '''
        data = self.data[colname].to_numpy()

        X = []
        for e in self.eps:
            diff = data[e:data.size:e] - data[:data.size-e:e]

            X.append(diff)

        X = np.array(X, dtype=object)
    
        return X
    

    def plot_x_diff(self):
        '''
        Plots the increments X(t) over the whole length of the dataset with delta_t of 
        1 day. 
        '''
        plt.plot(self.data.index[1:], self.X[0])
    

    def plot_x(self):
        '''
        Plots X(t) with increments of 1 day. 
        '''
        plt.plot(self.data.index[1:], np.cumsum(self.X[0]))
        
