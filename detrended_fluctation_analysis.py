import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from repos.multifractal.multifractal import Multifractal


class FluctuationAnalysis():
    def __init__(self, data, nu):
        self.data = data
        self.nu = nu
        self.N = data.size
        self.s = self.N / self.nu

    def split_data(self):
        self.data
        
    def rescaled_range(self):
        pass

