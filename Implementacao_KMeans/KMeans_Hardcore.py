import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score  # para avaliação

class KMeansClustering:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))
        
    def initialize_centroids_plus_plus(self, X):
        np.random.seed(42)  # Para reprodutibilidade
        centroids = []
        # Escolhe o primeiro centróide aleatoriamente
        centroids.append(X[np.random.randint(0, X.shape[0])])
        
        for _ in range(1, self.k):
            dist_sq = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
            prob = dist_sq / dist_sq.sum()
            cumulative_prob = np.cumsum(prob)
            r = np.random.rand()
            next_centroid = X[np.searchsorted(cumulative_prob, r)]
            centroids.append(next_centroid)
        
        return np.array(centroids)

    def fit(self, X, max_i=100):
        self.centroids = self.initialize_centroids_plus_plus(X)
        
        for _ in range(max_i):
            y = []
            for data_point in X:
                distancia = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_n = np.argmin(distancia)
                y.append(cluster_n)
            y = np.array(y)
            
            cluster_centros = []
            for i in range(self.k):
                indices = np.argwhere(y == i).flatten()
                if len(indices) == 0:
                    cluster_centros.append(self.centroids[i])
                else:
                    cluster_centros.append(np.mean(X[indices], axis=0))
            
            novos_centroids = np.array(cluster_centros)
            
            if np.max(np.abs(self.centroids - novos_centroids)) < 0.0001:
                break
            else:
                self.centroids = novos_centroids
                
        return y


# Carregando a base de Dados Iris
iris = load_iris()
X = iris.data  # apenas atributos (sem target)

# Experimento com k=3
modelo3 = KMeansClustering(k=3)
labels3 = modelo3.fit(X)
score3 = silhouette_score(X, labels3)

# Experimento com k=5
modelo5 = KMeansClustering(k=5)
labels5 = modelo5.fit(X)
score5 = silhouette_score(X, labels5)


# Resultados

print("=== Avaliacao dos Clusters com Kmeans Hardcore ===")
print(f"Silhouette Score com k=3: {score3:.4f}")
print(f"Silhouette Score com k=5: {score5:.4f}\n")


# Gráficos salvos como imagens

# Plot com 3 clusters
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 2], X[:, 3], c=labels3, cmap='viridis', s=30)
plt.scatter(modelo3.centroids[:, 2], modelo3.centroids[:, 3], c='red', marker='X', s=200)
plt.title(f"KMeans - 3 Clusters\nSilhouette: {score3:.4f}")
plt.savefig("clusters_3.png")
plt.close()

# Plot com 5 clusters
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 2], X[:, 3], c=labels5, cmap='viridis', s=30)
plt.scatter(modelo5.centroids[:, 2], modelo5.centroids[:, 3], c='red', marker='X', s=200)
plt.title(f"KMeans - 5 Clusters\nSilhouette: {score5:.4f}")
plt.savefig("clusters_5.png")
plt.close()
