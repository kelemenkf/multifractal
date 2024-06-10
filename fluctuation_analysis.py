import numpy as np
import matplotlib.pyplot as plt
import math

from repos.multifractal.rescaled_range import RescaledRange


class FluctuationAnalysis(RescaledRange):
    def __init__(self, data, b=2, nu_max=5):
        super().__init__(data, b)

        self.nu_min = math.ceil(math.log(10,self.b))
        if self.nu_min > nu_max:
            raise ValueError("self.nu has to be larger")
        self.nu_max = nu_max
        self.nu = np.array(range(self.nu_min,self.nu_max))
        self.s = self.N // (self.b**self.nu)


    def get_s(self):
        print(self.s)