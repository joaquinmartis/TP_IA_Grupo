import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import time
from substractive_clustering import substractive_clustering
from fis import *
import datetime as dt


def calcular_error_cuadratico_medio(datos, recta):
	return np.mean((datos - recta) ** 2)
	

def generar_rectas():
	pass

def main():
	csvFile = pd.read_csv('spy.csv')
	print(csvFile)
	csvFile.to_numpy()
	csvTranspuesto=csvFile.T
	datos_close=csvFile["Close"]
	datos_date=csvFile["Date"]
	datos_y=datos_close.to_numpy()
	datos_x_fechas= [dt.datetime.strptime(date,'%d/%m/%y').date() for date in datos_date.to_numpy()]
	datos_x=np.arange(len(datos_y))
	datos=np.vstack((datos_x, datos_y)).T
	fis = FIS()


	
	vec_reglas=[(0.3,0.9),(0.2,None),]
	Ra=1.1
	Rb=0
	label,n_clusters=fis.genFIS(datos,Ra,Rb=0)
	#Ra chico mas clusters
	fis.viewInputs()
	r = fis.evalFIS(np.vstack(datos_x))



	#vec_reglas=[(0.3,0.9),(0.2,0),(0.23,0),(0.4,0),(0.5,0),]
	#cant_reglas=1
	#plt.figure()
	#for i in range(cant_reglas):
	#	fis=FIS()
	#	regla=vec_reglas[i]
	#	label,n_clusters=fis.genFIS(datos, regla[0],regla[1])
	#	fis.viewInputs()
	#	r = fis.evalFIS(np.vstack(datos_x))
	#	
	#	plt.plot(datos_x_fechas,r,linestyle='--',color='r') #Recta Sugeno
	error_cuadratico_medio=[]
	cant_reglas=[]

	plt.figure()
	recta=np.vstack((datos_x, r)).T
	error_cuadratico_medio.append(calcular_error_cuadratico_medio(datos,recta))
	cant_reglas.append(3)
	plt.plot(cant_reglas,error_cuadratico_medio,'o')
	
	#plt.plot(datos_x,datos_y)

	plt.figure()
	#plt.plot(datos_x_fechas,datos_y) #Grafico de datos
	#colors = []
	#for _ in range(n_clusters):
	#	colors.append([random.random(), random.random(), random.random()])

	#Graficar los datos con colores aleatorios según el clúster
	#plt.figure(figsize=(8, 6))

	#for i in range(n_clusters):
	#	cluster_points = datos_x_fechas[label == i]
	#	print(cluster_points)
	#	
	#
	#	plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
	#for i in range(len(datos_x_fechas)):
	#	plt.scatter(datos_x_fechas[i], datos_y[i], color=colors[label[i]], label=f'Cluster {label[i]}' if i == 0 else "")


	print(label)
	plt.plot(datos_x_fechas,datos_y)
	plt.plot(datos_x_fechas,r,linestyle='--') #Recta Sugeno
	#plt.yscale('log') #Opcional
	plt.show()
	print(fis.solutions)


if __name__=="__main__":
	main()