import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import statsmodels.api as sm
from matplotlib.patches import Rectangle
import sys


from .multifractal import Multifractal


class DataHandler():
    def __init__(self, data):
        self.data = data


    