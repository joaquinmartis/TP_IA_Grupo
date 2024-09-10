import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from substractive_clustering import substractive_clustering
#from substractive_clustering_original import subclust2 as substractive_clustering
from reglas_fis import FISRule
from inputs_fis import FISInput
#def gaussmf(data, mean, sigma):
#        return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class FIS:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []


    
    def __gaussmf(self,data, mean, sigma):
        return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

    def genFIS(self, data, radii):

        start_time = time.time()
        #Clustering con los datos y clasificacion de cada punto a un cluster
        labels, cluster_center = substractive_clustering(data, radii)

        print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [FISInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    
    
    def entrenar(self, data):
        not_targets = data[:,:-1] #El ultimo elemento es el target. Agarra todos menos el ultimo
        targets = data[:,-1] #Agarra el ultimo elemento
        #P=not_targets


        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(self.__gaussmf(not_targets,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T

        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        not_targets = np.c_[not_targets, np.ones(len(not_targets))]
        n_vars = not_targets.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = not_targets[:, orden]


        A = acti*inp/sumMu

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = targets
        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        return 0

    def evalFIS(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(self.__gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        for input in self.inputs:
            input.view()
