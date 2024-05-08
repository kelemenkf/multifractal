import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import sys


class Multifractal():
    def __init__(self, b, M, support, E, k=0, mu=[1]):
        self.b = b
        if sum(M) == 1:
            self.M = np.array(M, dtype='f')
        else:
            raise ValueError("Multipliers do not add to 1")
        self.support = np.array([support], dtype='f')
        self.E = E
        self.k = k
        self.mu = np.array(mu)
        self.eps = self.b**(-self.k)


    def __str__(self):
        return f"{self.mu}"


    def set_eps(self):
        self.k += 1
        self.eps = self.b**(-self.k)


    def iterate(self, k):
        while k > 0:
            self.set_eps()
            temp = []
            for i in self.M:
                for j in self.mu:
                    temp.append(i*j)
            self.mu = np.array(temp)
            k -= 1


    def check_measure(self):
        return self.mu.sum()


    def plot_measure(self):
        fig, ax = plt.subplots()
        plot = ax.bar(np.linspace(0,1-(1/self.b**self.k),self.b**self.k),self.mu,1/(self.b**self.k),align='edge')
        plt.show()


    def animate(self, frames):
        fig, ax = plt.subplots()

        bars = ax.bar(0,1,1,align='edge')

        def animate_frame(frame):
            ax.clear()

            self.iterate(1)

            while self.k > len(bars):
                bars.append(ax.bar(0,1,1,align='edge'))

            for i, bar in enumerate(bars):
                bar.set_height(self.mu[i])
                bar.set_width(1/(self.b**self.k))

            return bars

        print(bars)

        ani = animation.FuncAnimation(fig=fig, func=animate_frame, frames=frames, interval=120)
        plt.show()
