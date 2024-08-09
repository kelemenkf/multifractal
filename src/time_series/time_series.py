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
        self.scale_iterator - designates the current index of the scale array self.scale_lengths.
        It is used in R/S, FA, DFA and MF_DFA to loop over the different scales and split the data
        at each scale. 
        '''
        self.data = data
        self.data_type = data_type
        if self.data_type == 'increments':
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
        self.scale_iterator = 0
        self.spl_data = self.split_data(self.data)
        self.spl_data_r = self.split_data(np.flip(self.data))
        self.time_index = np.array(range(len(self.data)))
        self.time_index_split = self.split_data(self.time_index)
        self.time_index_split_reverse = self.split_data(np.flip(self.time_index))

    
    def determine_nu_max(self):
        '''
        Determines the the upper limit of the the allowable iterations.
        '''
        MINIMUM_SCALE_LENGTH = 11
        return math.ceil(math.log(self.data_length // MINIMUM_SCALE_LENGTH, self.b))


    def determine_limits(self):
        '''
        Determines the limits of the allowable scales for the different methods.
        '''
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
        and returns it as a sefl.N_s[self.scale_iterator] x self.s matrix. If the size of the data,
        is not exactly equal to the number of segments times the size of the segments, 
        the residual data is left off.
        '''
        split_data = [data[i:i+self.scale_lengths[self.scale_iterator]] for i in range(0,self.data_length,self.scale_lengths[self.scale_iterator])]
        if self.N_s[self.scale_iterator] * self.scale_lengths[self.scale_iterator] != self.data_length:
            return np.array(split_data[:self.N_s[self.scale_iterator]])
        else:
            return np.array(split_data[:self.N_s[self.scale_iterator]])
        

    def reset_data(self):
        '''
        Splits both the original and the reversed series, as well as the index and the reversed index.
        '''
        self.spl_data = self.split_data(self.data)
        self.spl_data_r = self.split_data(np.flip(self.data))
        self.time_index_split = self.split_data(self.time_index)
        self.time_index_split_reverse = self.split_data(np.flip(self.time_index))
        

    def set_scale_iterator(self, iterator):
        '''
        Sets the iterator variable self.scale_iterator manually.
        '''
        self.scale_iterator = iterator
        self.reset_data()


    def shuffle_data(self):
        '''
        Shuffles the increment series of self.data.
        '''
        rng = np.random.default_rng()
        rng.shuffle(self.increments)
