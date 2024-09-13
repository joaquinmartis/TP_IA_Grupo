"""Subtractive Clustering Algorithm
"""
__author__ = 'Daniel Albornoz'


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

def substractive_clustering(datos, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
    """ 
    datos: matriz con los datos para trabajar
    Ra: Es un hiperparametro que determina el radio de pertenencia a un centro de cluster. Mayor Ra mayor radio, menor Ra manor radio
    Rb: Es un hiperparametro que determina cuanto potencial se le resta a cada punto "cercano" a un centro de cluster. Es una medida del "impacto" de cada centro de cluster al serle adjudicado potencial 0
        -Rb>Ra para que los centros de cluster no se encuentren mur cercanos
    AcceptRatio: Valor para algoritmo de aceptacion de clusters
    RejectRatio: Idem AcceptRatio
    """
    #Si el valor de Rb es por defecto o 0 se le asigna un valor para cumplir Rb>Ra
    if Rb==0:
        Rb = Ra*1.15

    alpha=(Ra/2)**2
    beta=(Rb/2)**2
    vec_centros = []
    #Normalizacion de los datos recibidos
    scaler = MinMaxScaler()
    scaler.fit(datos)
    datos_normalizados = scaler.transform(datos)
    #Calculo de potenciales de cada uno de los puntos
    matriz_distancias = distance_matrix(datos_normalizados,datos_normalizados)
    matriz_potenciales = np.sum(np.exp(-matriz_distancias**2/alpha),axis=0)
    #Calculo del punto incial con maximo potencial
    punto_mayor_potencial_actual=np.argmax(matriz_potenciales)
    punto_elegido_centro = datos_normalizados[punto_mayor_potencial_actual]
    potencial_actual=matriz_potenciales[punto_mayor_potencial_actual]
    vec_centros = [punto_elegido_centro]

    continuar=True
    restar_potencial = True

    while continuar:
        potencial_ant = potencial_actual

        #Resto los potenciales del centro elegido
        if restar_potencial:
            potenciales_desde_centro_elegido=[]
            for punto in datos_normalizados:
                potenciales_desde_centro_elegido.append(np.exp(-np.linalg.norm(punto-punto_elegido_centro)**2/beta))
            matriz_potenciales=matriz_potenciales-potencial_actual*np.array(potenciales_desde_centro_elegido)
        restar_potencial = True

        #Calculo del punto con maximo potencial
        punto_mayor_potencial_actual=np.argmax(matriz_potenciales)
        punto_elegido_centro = datos_normalizados[punto_mayor_potencial_actual]
        potencial_actual=matriz_potenciales[punto_mayor_potencial_actual]

        #Algoritmo para aceptar un nuevo candidato a centro de cluster -> Diapositiva 34 de clustering
        
        if potencial_actual>AcceptRatio*potencial_ant: #1
            vec_centros = np.vstack((vec_centros,punto_elegido_centro)) #Centro aceptado
        elif potencial_actual<RejectRatio*potencial_ant: #2
            continuar=False #Rechazo del centro y fin del algoritmo
        else: #3
            dr = np.min([np.linalg.norm(v-punto_elegido_centro) for v in vec_centros]) #Calcula la minima distancia entre el centro candidato(punto_elegido_centro) y el resto de los vec_centros
            if dr/Ra+potencial_actual/potencial_ant>=1: #Calculo para ver si el centro candidato es el elegido
                vec_centros = np.vstack((vec_centros,punto_elegido_centro))
            else:
                matriz_potenciales[punto_mayor_potencial_actual]=0
                restar_potencial = False

        #Verificar si existen puntos con potencial de ser vec_centros del nuevo cluster
        hay_potenciales_mayor_que_cero=any(v>0 for v in matriz_potenciales) #Podria optimizarse
        if continuar==True and not hay_potenciales_mayor_que_cero:
            continuar = False

    #Calculo de las distancias de cada punto a cada centro
    distancias=[]
    for centro in vec_centros:
        distancia_a_centro=[]
        for punto in datos_normalizados:
            dist=np.linalg.norm(punto-centro)
            distancia_a_centro.append(dist)
        distancias.append(distancia_a_centro)
    #Calculo del vector de pertenencia. Devuelve el centro al que pertenece cada punto
    vector_de_pertenencia = np.argmin(distancias, axis=0)
    vec_centros = scaler.inverse_transform(vec_centros)

    return vector_de_pertenencia, vec_centros
    
if __name__=="__main__":
    centro_1 = np.random.rand(25,2)+[1,1]
    centro_2 = np.random.rand(50,2)+[10,1.5]
    centro_3 = np.random.rand(20,2)+[4.9,5.8]
    datos = np.append(centro_1,centro_2, axis=0)
    datos = np.append(datos,centro_3, axis=0)

    vector_de_pertenencia,centros = substractive_clustering(datos,1)

    plt.figure()
    plt.scatter(datos[:,0],datos[:,1], c=vector_de_pertenencia)
    plt.scatter(centros[:,0],centros[:,1], marker='X')
    plt.show()
