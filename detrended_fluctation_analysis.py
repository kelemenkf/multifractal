import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from repos.multifractal.multifractal import Multifractal


class FluctuationAnalysis():
    def __init__(self, data, nu):
        self.data = data
        self.diff_data = np.diff(data)
        self.nu = nu
        self.N = self.diff_data.size
        self.s = self.N // self.nu


    def split_data(self):
        split_data = [self.diff_data[i:i+self.s] for i in range(0,self.N,self.s)]
        return np.array(split_data)
        

    def rescaled_range(self):
        pass

