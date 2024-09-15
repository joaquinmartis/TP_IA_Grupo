import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
import datetime as dt
from fis import *
from clustering import Clustering
from Sugeno import Sugeno


def leer_datos(ruta_archivo):
	data_file = pd.read_csv(ruta_archivo)
	data_file.to_numpy()
	datos_close=data_file["Close"]
	datos_date=data_file["Date"]
	datos_y=datos_close.to_numpy()
	datos_x_fechas= [dt.datetime.strptime(date,'%d/%m/%y').date() for date in datos_date.to_numpy()]
	datos_x = np.array([(fecha - datos_x_fechas[0]).days for fecha in datos_x_fechas], dtype=float)

	return datos_x, datos_y, datos_x_fechas

def obtener_datos_sobremuestreados(datos):
	datos_sobremuestreados = []
	for i in range(len(datos) - 1):
		datos_sobremuestreados.append(datos[i])
		datos_sobremuestreados.append((datos[i] + datos[i + 1]) / 2)
	datos_sobremuestreados.append(datos[-1])
	
	return datos_sobremuestreados

def calcular_error_cuadratico_medio(y_target, y_pred):
	return np.mean((y_target - y_pred) ** 2)

# Función para realizar holdout repetido
def holdout_repetido(X, y,Ra, n_splits=5, n_repeats=10, test_size=0.3):
	# RepeatedKFold: realiza n_splits splits y n_repeats repeticiones
	rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
	resultados = []
	for train_index, test_index in rkf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		datos=np.vstack((X_train, y_train)).T

		# Crear y entrenar el FIS
		clustering_method = Clustering(datos)
		clustering_method.substractive(Ra)
		sgno = Sugeno()
		sgno.generar_fis(datos, clustering_method.vec_reglas)
        
        # Evaluar el FIS en el conjunto de prueba
		y_pred = sgno.evalFIS(np.vstack(X_test))
        
        # Calcular error (por ejemplo, RMSE)
		rmse = calcular_error_cuadratico_medio(y_pred,np.array(y_test))
		resultados.append(rmse)
    
    # Promediar el error en todas las repeticiones
	promedio_error = np.mean(resultados)
	return promedio_error

def main():
	
	datos_x, datos_y, datos_x_fechas = leer_datos('spy.csv')

	# Graficar los datos
	plt.figure(figsize=(12, 6))
	plt.plot(datos_x_fechas, datos_y , label='Precio de Cierre', color='b')
	# Configurar etiquetas y título
	plt.xlabel('Fecha')
	plt.ylabel('Precio de Cierre')
	plt.title('Precio de Cierre de las Acciones del S&P 500')
	plt.legend()
	plt.grid(True)
	plt.show()	
	
	datos=np.vstack((datos_x, datos_y)).T
	
	vec_reglas=[
		
		{"Ra":0.20},
		{"Ra":0.2},
		{"Ra":0.4},
		{"Ra":0.6},
		{"Ra":1.1},
		# {"Ra":1, "Rb":0},
		# {"Ra":0.2, "Rb":0.21},
		# {"Ra":0.4, "Rb":0.42},
		# {"Ra":0.6, "Rb":0.63},
		# {"Ra":0.8, "Rb":0.84},
	]

	MSE_Resustitucion=[]
	MSE_HoldOutRepetido=[]
	modelos=[]

	for regla in vec_reglas:
		
		clustering_method = Clustering(datos)
		print("vuelta 1")
		### KMeans
		# clustering_method.kmeans(datos, 3)
		## Subtractive
		clustering_method.substractive(regla["Ra"])
		sgno = Sugeno()
		sgno.generar_fis(datos, clustering_method.vec_reglas)
		r = sgno.evalFIS(np.vstack(datos_x))
		MSE_Resustitucion.append(calcular_error_cuadratico_medio(np.array(datos_y), r))
		MSE_HoldOutRepetido.append(holdout_repetido(datos_x, datos_y,regla["Ra"]))
		print("vuelta 1.1")
		modelos.append(sgno)
		
	
	#Graficar el error cuadratico medio (x=radioA y=error cuadratico medio)
	plt.plot([regla["Ra"] for regla in vec_reglas], MSE_Resustitucion)
	plt.xlabel("Ra")
	plt.ylabel("Error cuadrático medio Resustitución")
	plt.title("Mse de Resustitucion vs Ra")
	plt.show(block=False)

	plt.plot([regla["Ra"] for regla in vec_reglas], MSE_HoldOutRepetido)
	plt.xlabel("Ra")
	plt.ylabel("Error cuadrático medio Hold Out Repetido")
	plt.title("Mse Hold Out Repetido vs Ra")
	plt.show()

	indice_mejor_modelo = np.argmin(MSE_HoldOutRepetido)
	mejor_modelo=modelos[indice_mejor_modelo]
	# Sobremuestrear los datos originales
	datos_sobremuestreados_y = obtener_datos_sobremuestreados(datos_y)
	datos_sobremuestreados_x = obtener_datos_sobremuestreados(datos_x)
	mejor_modelo.viewInputs()
	# evaluar modelo Sugeno
	r = mejor_modelo.evalFIS(np.vstack(datos_sobremuestreados_x))
	plt.figure()
	plt.plot([datos_x_fechas[0] + pd.to_timedelta(d, unit='D') for d in datos_sobremuestreados_x], datos_sobremuestreados_y, color='blue')
	plt.plot([datos_x_fechas[0] + pd.to_timedelta(d, unit='D') for d in datos_sobremuestreados_x], r, '--', color='red')
	plt.xlabel("Años")
	plt.ylabel("Valor de cierre")
	plt.show()

	#extrapolacion es hacer un for creando datos desde el ultimo dato conocido... se puede poner una barra para identificar donde comienza
	input("Presione enter para finalizar")	


if __name__=="__main__":
	main()