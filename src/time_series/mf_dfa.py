import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from multifractal.time_series.detrended_fluctuation_analysis import DFA

class MF_DFA(DFA):
    def __init__(self, data, b=2, method='mf_dfa', m=2, data_type='diff', q=[-5,5], gran=0.1, modified=False, nu_max=8):
        super().__init__(data, b=b, method=method, m=m, data_type=data_type, nu_max=nu_max)

        '''
        self.data_type - assumes that the time series is not integrated, but the TimeSeries base class
        integrates it. 
        self.modified - If True it takes the sum of the profile. This will lead to better estimations of h(q) near
        h(q) = 0.
        '''
        self.q_range = np.linspace(q[0],q[-1],int((q[1]-q[0])/gran)+1)
        if modified:
            self.data = self.modified_data()
            self.reset_data()
        self.fa_q = self.fluctuation_functions()
        self.h_q = self.calc_h_q()


    def modified_data(self):
        '''
        Returns the modified data for the modified MF-DFA which is more accurate 
        near h(q) = 0.
        '''
        data = np.cumsum(self.data - np.mean(self.data))
        return data


    def q_fluctuation_0(self):
        '''
        Calculates the fluctuation function with q = 0.
        '''
        f_2 = np.concatenate((self.squared_fluctuation(self.spl_data, self.x_split), self.squared_fluctuation(self.spl_data_r, self.x_split_r)))
        fa_0 = np.exp(np.sum(np.log(f_2)) / (self.N_s[self.i] * 4))
        return fa_0


    def q_fluctuation(self, q=[-5,5], gran=0.1):
        '''
        Obtain the qth order fluctuation function.
        '''
        fa = []
        fa_0 = self.q_fluctuation_0()
        for q in self.q_range:
            if q == 0:
                fa.append(fa_0)
            else:
                f_2 = np.concatenate((self.squared_fluctuation(self.spl_data, self.x_split), self.squared_fluctuation(self.spl_data_r, self.x_split_r)))
                fa_q = np.mean((f_2**(q/2)))**(1/q)
                fa.append(fa_q)
        return fa


    def fluctuation_functions(self):
        '''
        Calculates the fluctation function for different scales s, and for all orders q
        at each scale.
        '''
        fa_q = [self.q_fluctuation()]
        for n in range(1, self.nu.size):
            self.i += 1
            self.reset_data()
            fa_q.append(self.q_fluctuation())
        self.i = 0
        self.reset_data()
        return np.array(fa_q)
    

    def plot_f_q_s(self, q):
        '''
        Function to individually plot the data and the best fitted line on a 
        logarithmic scale of F_q(s) for a given q, specified by the parameter. 
        '''
        q_index = np.where(np.round(self.q_range,1) == q)
        print(q_index)
        data = self.fa_q[:, q_index]
        y = np.log(data)
        x = np.log(self.s)
        params = self.get_slope(y, x)

        plt.scatter(x, y)
        print(params)
        best_fit_y = params[0] + (params[1] * x)

        plt.plot(x, best_fit_y)
        plt.legend(f"{q}")


    def get_slope(self, y, x):
        '''
        Calculates slope of an ols regression of x on y. Constant is added. 
        '''
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        return results.params
    

    def calc_h_q(self):
        '''
        Calculates the h(q) function the generalized Hurst exponetns, describing the 
        scaling behaviour of the fluctuations. 
        '''
        h_q = {}
        data = self.fa_q
        for q in range(data.shape[1]):
            h = self.get_slope(np.log(data[:, q]), np.log(self.s))[1]
            h_q.update({self.q_range[q]:h})
        return h_q
    

    def plot_h_q(self):
        '''
        Plots the h(q) function.
        '''
        plt.plot(list(self.h_q.keys()), list(self.h_q.values()))
        plt.xlabel("$q$")
        plt.ylabel("$h(q)$")


    def plot_fa(self, Q=list(range(-5,6))):
        '''
        Plots the fluctuation functions for the different qs. 
        '''
        fa_q = self.fluctuation_functions()
        for q in range(self.q_range.size):
            plt.plot(np.log(self.s), np.log(fa_q[:,q]), label=[f"{q}"])


    def scaling_function(self):
        tau = np.array(list(self.h_q.keys())) * np.array(list(self.h_q.values())) - 1
        plt.plot(self.h_q.keys(), tau)
        plt.xlabel("$q$")
        plt.ylabel("τ(q)")


    #TODO spectrum



