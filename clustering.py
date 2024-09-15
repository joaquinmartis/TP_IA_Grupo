from substractive_clustering import substractive_clustering

class Clustering:
    def __init__(self, data):
        self.data=data
        self.cant_reglas=None
        self.vec_reglas=None
        self.vec_pertenencia=None

    def kmeans(self, data, cant_clusters):
        """
        data: matriz con los datos para trabajar
        cant_clusters: cantidad de clusters a formar
        """
        from sklearn.cluster import KMeans
        self.cant_reglas=cant_clusters
        kmeans = KMeans(n_clusters=cant_clusters, random_state=0).fit(data)
        self.vec_pertenencia=kmeans.labels_
        self.vec_reglas=kmeans.cluster_centers_

    def substractive(self, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
        """
        Ra: Es un hiperparametro que determina el radio de pertenencia a un centro de cluster. Mayor Ra mayor radio, menor Ra manor radio
        Rb: Es un hiperparametro que determina cuanto potencial se le resta a cada punto "cercano" a un centro de cluster. Es una medida del "impacto" de cada centro de cluster al serle adjudicado potencial 0
            -Rb>Ra para que los centros de cluster no se encuentren mur cercanos
        AcceptRatio: Valor para algoritmo de aceptacion de clusters
        RejectRatio: Idem AcceptRatio
        """
        self.vec_pertenencia, self.vec_reglas  = substractive_clustering(self.data, Ra, Rb, AcceptRatio, RejectRatio)
        self.cant_reglas=len(self.vec_reglas)