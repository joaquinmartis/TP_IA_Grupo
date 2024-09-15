import numpy as np

class Sugeno:
    def __init__(self):
        self.memberfunc = []
        self.rules = []
    
    def __gaussmf(self,data, mean, sigma):
        return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

    def generar_funciones_pertenencia(self, data, cluster_center):
        """
        data: datos de entrada
        cluster_center: media de la funciones de pertenencia gaussianas
        """
        self.rules = cluster_center
        P = data[:,:-1] #El ultimo elemento es el target. Agarra todos menos el ultimo
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)
        sigma = np.array([(maxValue[i]-minValue[i])/np.sqrt(8) for i in range(len(maxValue))])
        self.memberfunc = [np.prod(self.__gaussmf(data,cluster,sigma),axis=1) for cluster in cluster_center]

    
    
    def entrenar(self, data):
        not_targets = data[:,:-1] #El ultimo elemento es el target. Agarra todos menos el ultimo
        targets = data[:,-1] #Agarra el ultimo elemento
        
        nivel_acti = np.array(self.memberfunc).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1)) #Para cada dato suma los valores de verdad de cada cluster

        not_targets_with_ones = np.c_[not_targets, np.ones(len(not_targets))]
        n_vars = not_targets_with_ones.shape[1] #Cantidad de variables

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = not_targets_with_ones[:, orden] #Es una matriz con los datos de entrada repetidos para cada cluster

        A = acti*inp/sumMu
        b = targets

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)

    def evalFIS(self, data):
 
        nivel_acti = np.array(self.memberfunc).T
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
