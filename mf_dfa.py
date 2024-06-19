import numpy as np

from repos.multifractal.detrended_fluctuation_analysis import DFA

class MF_DFA(DFA):
    def __init__(self, data, b=2, method='mf_dfa', m=2, data_type='diff'):
        super().__init__(data, b=b, method=method, data_type=data_type)

        self.m = m


    def q_fluctuation(self, q=[-5,5], gran=0.1):
        '''
        Obtain the qth order fluctuation function
        '''
        q_range = np.linspace(q[0],q[-1],int((q[1]-q[0])/gran)+1)
        fa = []
        for q in q_range:
            f_2 = np.concatenate((self.squared_fluctuation(self.spl_data, self.x_split), self.squared_fluctuation(self.spl_data_r, self.x_split_r)))
            fa_q = np.mean((f_2**(q/2)))**(1/q)
            fa.append(fa_q)
        return fa


    def q_fluctuation_0(self):
        '''
        Calculates the fluctuation function with q = 0.
        '''
        f_2 = np.concatenate((self.squared_fluctuation(self.spl_data, self.x_split), self.squared_fluctuation(self.spl_data_r, self.x_split_r)))
        fa_0 = np.exp(np.sum(np.log(f_2)) / self.N_s[self.i] * 4)


    def fluctuation_functions(self):
        '''
        Calculates the fluctation function for different scales s, and for all orders q
        at each scale.
        '''
        fa_q = [self.q_fluctuation]
        for n in range(1, self.nu.size):
            print(self.i)
            self.i += 1
            self.reset_data()
            fa_q.append(self.q_fluctuation())
        self.i = 0
        self.reset_data()
        return fa_q



