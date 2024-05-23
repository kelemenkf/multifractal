import numpy as np
import stochastic

from .multifractal import Multifractal

class Simulator(Multifractal):
    def __init__(self, b, M, support_endpoints, E=1, k=0, mu=[1], P=[], r_type="", scale=1, loc=0):
        super().__init__(b, M, support_endpoints, E, k, mu, P, r_type, scale, loc)



