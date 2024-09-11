# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:56:16 2020

@author: Daniel Albornoz

Implementación similar a genfis de Matlab.
Sugeno type FIS. Generado a partir de clustering substractivo.

"""
__author__ = 'Daniel Albornoz'

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import time
from substractive_clustering import substractive_clustering
from fis import *

def main1():

    data_x = np.arange(-10,10,0.1)
    data_y = -0.5*data_x**3-0.6*data_x**2+10*data_x+1 #my_exponential(9, 0.5,1, data_x)

    plt.plot(data_x, data_y)
    # plt.ylim(-20,20)
    plt.xlim(-7,7)

    data = np.vstack((data_x, data_y)).T

    fis = FIS()
    fis.genFIS(data, 1.1)
    fis.viewInputs()
    r = fis.evalFIS(np.vstack(data_x))

    plt.figure()
    plt.plot(data_x,data_y)
    plt.plot(data_x,r,linestyle='--')
    plt.show()
    fis.solutions

    # r1 = data_x*-2.29539539+ -41.21850973
    # r2 = data_x*-15.47376916 -79.82911266
    # r3 = data_x*-15.47376916 -79.82911266
    # plt.plot(data_x,r1)
    # plt.plot(data_x,r2)
    # plt.plot(data_x,r3)

def main():
    datos=np.loadtxt("diodo.txt", dtype='f', delimiter='\t')
    #print(datos)
    #datos_x=datos[:,0]
    #datos_y=datos[:,1]

    sorted_data=np.sort(datos,axis=0)
    #print(sorted_data)
    datos_x=sorted_data[:,0]
    datos_y=sorted_data[:,1]
    
    #plt.plot(datos_x, datos_y)
    #plt.show()

    fis = FIS()
    radio=1.1
    fis.genFIS(sorted_data, radio)
    fis.viewInputs()
    r = fis.evalFIS(np.vstack(datos_x))

    plt.figure()
    plt.plot(datos_x,datos_y)
    plt.plot(datos_x,r,linestyle='--')
    plt.show()
    print(fis.solutions)



if __name__=="__main__":
    main()
