import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Carregar base de Dados Iris
iris = load_iris()
X = iris.data

# KMeans com k=3 (melhor k)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

score = silhouette_score(X, labels)
print("=== Avaliacao com sklearn KMeans ===")
print(f"Melhor numero de clusters: k={k}")
print(f"Silhouette Score: {score:.4f}")
print("=" * 40 + "\n")

# PCA com 1 componente
pca1 = PCA(n_components=1)
X_pca1 = pca1.fit_transform(X)

centroids_pca1 = pca1.transform(kmeans.cluster_centers_)

plt.figure(figsize=(7, 4))
plt.scatter(X_pca1[:, 0], np.zeros_like(X_pca1[:, 0]), c=labels, cmap='viridis', s=40)
plt.scatter(centroids_pca1[:, 0], np.zeros_like(centroids_pca1[:, 0]),
            c='red', marker='X', s=200, label="Centróides")
plt.title("Clusterização com PCA (1 Componente)")
plt.xlabel("Componente Principal 1")
plt.legend()
plt.savefig("pca_1_component.png")
plt.close()

# PCA com 2 componentes
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X)

centroids_pca2 = pca2.transform(kmeans.cluster_centers_)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels, cmap='viridis', s=40)
plt.scatter(centroids_pca2[:, 0], centroids_pca2[:, 1],
            c='red', marker='X', s=200, label="Centróides")
plt.title("Clusterização com PCA (2 Componentes)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.savefig("pca_2_components.png")
plt.close()