import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class FISInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids

    
    def __gaussmf(self,data, mean, sigma):
        return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

    def view(self, fecha):
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = self.__gaussmf(x,m,s)
            plt.plot([fecha+ pd.to_timedelta(d, unit='D') for d in x],y)
        plt.xlabel("Años")
        plt.ylabel("Valor de verdad")
