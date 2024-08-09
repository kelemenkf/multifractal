import numpy as np
import math

class TimeSeries():
    def __init__(self, data, b=2, method='dfa', data_type='profile', nu_max=None) -> None:
        '''
        self.data_type - wether the time series is integrated or not. Default if that it is.
        self.b - the scalar with which the data is scaled. E.g. if b = 2, the scales considred are always 
        half that of the previous ones. 
        self.nu - the number of different scales considered in the analysis
        self.s - the lengths of the segments at each scale
        self.N_s - the number of segments at each scale 
        self.i - an interator variable, designating the index of the current scale array self.s.
        It is used in R/S, FA and DFA to loop over the different scales and split the data
        at each scale. 
        '''
        self.data = data
        self.data_type = data_type
        if self.data_type == 'increment':
            self.data = np.insert(self.data, 0, 0)
            self.data = np.cumsum(self.data)
            self.data_type == 'profile'
        self.method = method
        self.increments = np.diff(self.data)
        self.increments_reverse = np.flip(self.increments)
        self.b = b
        self.data_length = self.increments.size
        self.nu_max = nu_max
        if self.nu_max == None:
            self.nu_max = self.determine_nu_max()
        self.nu = self.determine_limits()
        self.N_s = np.array(self.b**self.nu).astype(int)
        self.scale_lengths = self.data_length // self.N_s
        self.i = 0
        self.spl_data = self.split_data(self.data)
        self.spl_data_r = self.split_data(np.flip(self.data))
        self.time_index = np.array(range(len(self.data)))
        self.time_index_split = self.split_data(self.time_index)
        self.time_index_split_reverse = self.split_data(np.flip(self.time_index))

    
    def determine_nu_max(self):
        MINIMUM_SCALE_LENGTH = 11
        return math.ceil(math.log(self.data_length // MINIMUM_SCALE_LENGTH, self.b))


    def determine_limits(self):
        if self.method == 'fa':
            self.nu_min = math.ceil(math.log(10,self.b))
        elif self.method == 'dfa' or self.method == 'mf_dfa':
            self.nu_min = math.ceil(math.log(4,self.b))
        elif self.method == 'rs':
            return np.array(range(0, self.nu_max))
        return np.array(range(self.nu_min,self.nu_max))


    def split_data(self, data):
        '''
        Splits a time series of differences into self.nu, equidistant ranges,
        and returns it as a sefl.N_s[self.i] x self.s matrix. If the size of the data,
        is not exactly equal to the number of segments times the size of the segments, 
        the residual data is left off.
        '''
        split_data = [data[i:i+self.scale_lengths[self.i]] for i in range(0,self.data_length,self.scale_lengths[self.i])]
        if self.N_s[self.i] * self.scale_lengths[self.i] != self.data_length:
            return np.array(split_data[:self.N_s[self.i]])
        else:
            return np.array(split_data[:self.N_s[self.i]])
        

    def reset_data(self):
        '''
        Redoes the splitting of the data with the current value of self.i.
        '''
        self.spl_data = self.split_data(self.data)
        self.spl_data_r = self.split_data(np.flip(self.data))
        self.time_index_split = self.split_data(self.time_index)
        self.time_index_split_reverse = self.split_data(np.flip(self.time_index))
        

    def set_i(self, i):
        '''
        Sets the iterator variable self.i for inspection of the estimation at 
        different scales wihtout looping.
        '''
        self.i = i
        self.reset_data()


    def shuffle_data(self):
        rng = np.random.default_rng()
        rng.shuffle(self.diff_data)
        self.data = np.insert(self.data, 0, 0)
        self.data = np.cumsum(self.data)
        self.reset_data()
