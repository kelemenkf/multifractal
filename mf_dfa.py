import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from repos.multifractal.detrended_fluctuation_analysis import DFA

class MF_DFA(DFA):
    def __init__(self, data, b=2, method='mf_dfa', m=2, data_type='diff', q=[-5,5], gran=0.1):
        super().__init__(data, b=b, method=method, m=m, data_type=data_type)

        self.q_range = np.linspace(q[0],q[-1],int((q[1]-q[0])/gran)+1)
        self.h_q = self.calc_h_q()


    def q_fluctuation_0(self):
        '''
        Calculates the fluctuation function with q = 0.
        '''
        f_2 = np.concatenate((self.squared_fluctuation(self.spl_data, self.x_split), self.squared_fluctuation(self.spl_data_r, self.x_split_r)))
        fa_0 = np.exp(np.sum(np.log(f_2)) / (self.N_s[self.i] * 4))
        return fa_0


    def q_fluctuation(self, q=[-5,5], gran=0.1):
        '''
        Obtain the qth order fluctuation function
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


    def get_slope(self, y, x):
        '''
        Calculates slope of an ols regression of x on y. Constant is added. 
        '''
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        return results.params[1]
    

    def calc_h_q(self):
        '''
        Calculates the h(q) function the generalized Hurst exponetns, describing the 
        scaling behaviour of the fluctuations. 
        '''
        h_q = {}
        data = self.fluctuation_functions()
        for s in range(data.shape[1]):
            h = self.get_slope(np.log(data[:, s]), np.log(self.s))
            h_q.update({self.q_range[s]:h})
        return h_q
    

    def plot_h_q(self):
        '''
        Plots the h(q) function
        '''
        plt.plot(self.h_q.keys(), self.h_q.values())
        plt.xlabel("$q$")
        plt.ylabel("$h(q)$")


    def plot_fa(self, Q=list(range(-5,6))):
        '''
        Plots the fluctuation functions for the different qs. 
        '''
        fa_q = self.fluctuation_functions()
        for q in Q:
            print(q)
            plt.plot(self.s, fa_q[:,q])




