import numpy as np
from inputs_fis import FISInput
class Sugeno:
    def __init__(self):
        self.memberfunc = []
        self.rules = []
        self.inputs=[]
    
    def __gaussmf(self,data, mean, sigma):
        return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

    def generar_fis(self, data, centros):
        cluster_center = centros
        n_clusters = len(cluster_center)
        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)
        self.inputs = [FISInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    
    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(self.__gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]
        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        A = acti*inp/sumMu
        b = T
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

    def viewInputs(self,fecha):
        for input in self.inputs:
            input.view(fecha)

    def get_rules(self):
        return self.rules
