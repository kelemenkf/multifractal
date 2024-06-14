import numpy as np

class Stationary():
    def __init__(self, data, b, nu=5) -> None:
        '''
        self.N - the length of the data
        self.b - the scalar with which the data is scaled. E.g. if b = 2, the scales considred are always 
        half that of the previous ones. 
        self.nu - the number of different scales considered in the analysis
        self.s - the lengths of the segments at each scale
        self.i - an interator variable, designating the index of the current scale array self.s.
        It is used in R/S, FA and DFA to loop over the different scales and split the data
        at each scale. 
        self.x - the index of the time series data, to be used in polyfit
        '''
        self.data = data
        self.diff_data = np.diff(self.data)
        self.diff_data_r = np.flip(self.diff_data)
        self.b = b
        self.nu = np.array(list(range(0,nu+1)))
        self.N = self.diff_data.size
        self.s = self.N // (self.b**self.nu)
        self.i = 0
        self.spl_data = self.split_data(self.data)
        self.spl_data_r = self.split_data(np.flip(self.data))
        self.x = np.array(range(len(self.data)))
        self.x_split = self.split_data(self.x)

    
    def split_data(self, data):
        '''
        Splits a time series of differences into self.nu, equidistant ranges,
        and returns it as a sefl.b**self.nu x self.s matrix. 
        '''
        split_data = [data[i:i+self.s[self.i]] for i in range(0,self.N,self.s[self.i])]
        if self.b**self.nu[self.i] * self.s[self.i] != self.N:
            return np.array(split_data[:self.b**self.nu[self.i]])
        else:
            return np.array(split_data[:self.b**self.nu[self.i]])