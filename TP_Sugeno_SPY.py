import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fis import *
import datetime as dt
from clustering import Clustering
from Sugeno import Sugeno

def calcular_error_cuadratico_medio(datos, recta):
	return np.mean((datos - recta) ** 2)

def generar_rectas():
	pass

def leer_datos(ruta_archivo):
	data_file = pd.read_csv(ruta_archivo)
	data_file.to_numpy()
	datos_close=data_file["Close"]
	datos_date=data_file["Date"]
	datos_y=datos_close.to_numpy()
	datos_x_fechas= [dt.datetime.strptime(date,'%d/%m/%y').date() for date in datos_date.to_numpy()]
	datos_x=np.arange(len(datos_y))

	return datos_x, datos_y, datos_x_fechas

# def normalizar_datos(datos):
	# scaler = MinMaxScaler(feature_range=(0, 1))
	# datos_normalizados = scaler.fit_transform(datos)
	# return datos_normalizados

def obtener_datos_sobremuestreados(datos):
	datos_sobremuestreados = []
	for i in range(len(datos) - 1):
		datos_sobremuestreados.append(datos[i])
		datos_sobremuestreados.append((datos[i] + datos[i + 1]) / 2)
	datos_sobremuestreados.append(datos[-1])
	
	return datos_sobremuestreados

def graficar_error_cuadratico_medio_vs_cantidad_reglas(error_cuadratico_medio, cantidad_reglas):
	plt.plot(cantidad_reglas, error_cuadratico_medio, 'o-')
	plt.xlabel("Cantidad de reglas")
	plt.ylabel("Error cuadrático medio")
	plt.title("Error cuadrático medio vs cantidad de reglas")
	plt.show()

def entrenar_modelo_sugeno(datos, reglas, modelo):
	pass

def main():
	
	datos_x, datos_y, datos_x_fechas = leer_datos('spy.csv')
	datos=np.vstack((datos_x, datos_y)).T
	
	vec_reglas=[
		{"Ra":0.2, "Rb":0.21},
		{"Ra":0.4, "Rb":0.405},
		{"Ra":0.6, "Rb":1},
		{"Ra":1.1, "Rb":1.12},
		# {"Ra":1, "Rb":0},
		# {"Ra":0.2, "Rb":0.21},
		# {"Ra":0.4, "Rb":0.42},
		# {"Ra":0.6, "Rb":0.63},
		# {"Ra":0.8, "Rb":0.84},
	]

	errores_cuadraticos_medios=[]
	cant_reglas=[]

	for regla in vec_reglas:
		clustering_method = Clustering(datos)
		### KMeans
		# clustering_method.kmeans(datos, 3)
		## Subtractive
		clustering_method.substractive(regla["Ra"], regla["Rb"])
		sgno = Sugeno()
		sgno.generar_funciones_pertenencia(datos, clustering_method.vec_reglas)
		sgno.entrenar(datos)
		r = sgno.evalFIS(datos_x)
		plt.figure()
		plt.plot(datos_x, datos_y, '.', color='blue')
		plt.plot(datos_x, r, '-', color='red')

		# Sobremuestrear los datos originales
		datos_sobremuestreados_y = obtener_datos_sobremuestreados(datos_y)
		datos_sobremuestreados_x = np.arange(len(datos_sobremuestreados_y))
		datos_sobremuestreados=np.vstack((datos_sobremuestreados_x, datos_sobremuestreados_y)).T	

		# evaluar modelo Sugeno
		sgno = Sugeno()
		sgno.generar_funciones_pertenencia(datos_sobremuestreados, clustering_method.vec_reglas)
		sgno.entrenar(datos_sobremuestreados)
		r2 = sgno.evalFIS(datos_sobremuestreados_x)
		plt.figure()
		plt.plot(datos_sobremuestreados_x, datos_sobremuestreados_y, '.', color='blue')
		plt.plot(datos_sobremuestreados_x, r2, '-', color='red')
		plt.show()
		
		recta1=np.vstack((datos_x, r)).T
		recta2=np.vstack((datos_sobremuestreados_x, r2)).T
		
		errores_cuadraticos_medios.append(calcular_error_cuadratico_medio(datos, recta1))
		cant_reglas.append(len(clustering_method.vec_reglas))
		graficar_error_cuadratico_medio_vs_cantidad_reglas(errores_cuadraticos_medios, cant_reglas)
		
		errores_cuadraticos_medios.clear()
		cant_reglas.clear()
		
		errores_cuadraticos_medios.append(calcular_error_cuadratico_medio(datos_sobremuestreados, recta2))
		cant_reglas.append(len(clustering_method.vec_reglas))
		graficar_error_cuadratico_medio_vs_cantidad_reglas(errores_cuadraticos_medios, cant_reglas)


	# Graficar errores cuadraticos medios vs cantidad de reglas unidos por una linea
	# graficar_error_cuadratico_medio_vs_cantidad_reglas(errores_cuadraticos_medios, cant_reglas)
	
	# # Sobremuestrear los datos originales
	# datos_sobremuestreados_y = obtener_datos_sobremuestreados(datos_y)
	# datos_sobremuestreados_x = np.arange(len(datos_sobremuestreados_y))
	# datos_sobremuestreados=np.vstack((datos_sobremuestreados_x, datos_sobremuestreados_y)).T	

	# # evaluar modelo Sugeno
	# sgno = Sugeno()
	# sgno.generar_funciones_pertenencia(datos_sobremuestreados, clustering_method.vec_reglas)
	# sgno.entrenar(datos_sobremuestreados)
	# r = sgno.evalFIS(datos_sobremuestreados_x)
	# plt.figure()
	# plt.plot(datos_sobremuestreados_x, datos_sobremuestreados_y, '.', color='blue')
	# plt.plot(datos_sobremuestreados_x, r, '-', color='red')
	# plt.show()
	# recta=np.vstack((datos_x, r)).T
	# errores_cuadraticos_medios.append(calcular_error_cuadratico_medio(datos, recta))
	# cant_reglas.append(len(clustering_method.vec_reglas))
	# graficar_error_cuadratico_medio_vs_cantidad_reglas(errores_cuadraticos_medios, cant_reglas)




if __name__=="__main__":
	main()