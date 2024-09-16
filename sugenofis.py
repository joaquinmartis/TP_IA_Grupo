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
import pandas as pd
import pandas as pd
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

def main2():
    from matplotlib import cm

    x=np.linspace(-10,10,50)
    X,Y = np.meshgrid(x,x)
    Z = X**2+Y**2

    data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    fis3 = FIS()
    fis3.genFIS(data,1.2)
    fis3.viewInputs()

    r = fis3.evalFIS(np.vstack((X.ravel(), Y.ravel())).T)
    r = np.reshape(r, X.shape)


    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d') #fig.gca(projection='3d')
    surf = ax.plot(X,Y,Z, cmap=cm.Blues,
        linewidth=0, antialiased=False, alpha=0.3)

    surf = ax.plot(X,Y, r, cmap=cm.Reds,
        linewidth=0, antialiased=False, alpha=0.8)




def sugenoSPY():
    # Cargar los datos del archivo CSV
    spy = pd.read_csv('spy.csv') 
    
    # Convertir la columna 'Date' a formato de fecha
    spy['Date'] = pd.to_datetime(spy['Date'], format='%d/%m/%y')
    
    # Obtener la primera fecha en el DataFrame
    primera_fecha = spy['Date'].min()
    
    x= np.array((pd.to_datetime(spy['Date'], format='%d/%m/%y') - primera_fecha) // pd.Timedelta('1D'),dtype=float)
    y= np.array(spy['Close'],dtype=float)
    
 
    data= np.vstack((x,y)).T

    # Graficar los datos
    plt.figure(figsize=(12, 6))
    plt.plot(spy['Date'], spy['Close'] , label='Precio de Cierre', color='b')
    # Configurar etiquetas y título
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.title('Precio de Cierre de las Acciones del S&P 500')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    fis=FIS()
    fis.genFIS(data,0.6)
    fis.viewInputs()
    r=fis.evalFIS(np.vstack(x))
    plt.figure()
    plt.plot([primera_fecha + pd.to_timedelta(d, unit='D') for d in x],y)
    plt.plot([primera_fecha + pd.to_timedelta(d, unit='D') for d in x],r,linestyle='--')
    plt.show(block=False)
    input("Presione enter para finalizar")
if __name__=="__main__":
    sugenoSPY()