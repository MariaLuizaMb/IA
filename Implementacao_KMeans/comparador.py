import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from KMeans_Hardcore import KMeansClustering
from KMeans_Sk import KMeans


# Função para avaliar algoritmos
def avaliar_algoritmo(algoritmo, X, k, nome="Algoritmo"):
    inicio = time.perf_counter()

    # Verifica qual é o algoritmo para chamar o método correto
    if isinstance(algoritmo, KMeansClustering):
        labels = algoritmo.fit(X)  # já retorna os labels
    else:
        labels = algoritmo.fit_predict(X)  # sklearn retorna labels com fit_predict

    fim = time.perf_counter()

    # Métrica de qualidade
    score = silhouette_score(X, labels)

    # Impressão formatada
    print("=" * 40)
    print(f"Resultados - {nome}")
    print(f"k = {k}")
    print(f"Silhouette Score: {score:.4f}")
    print(f"Tempo de execucao: {fim - inicio:.4f} segundos")

    # Distribuição dos clusters
    unique, counts = np.unique(labels, return_counts=True)
    print("\nDistribuicao dos clusters:")
    for u, c in zip(unique, counts):
        print(f"   - Cluster {u}: {c} elementos")

    print("=" * 40 + "\n")

    return labels, score


# Carregando Dataset Iris
iris = load_iris()
X = iris.data


# Testar algoritmos
# Sklearn KMeans -> apenas k=3
print("\n##### Avaliacao Sklearn KMeans (k = 3) #####")
sk_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
avaliar_algoritmo(sk_kmeans, X, 3, nome="Sklearn KMeans")

# Hardcore KMeans -> k=3 e k=5
for k in [3, 5]:
    print(f"\n##### Avaliacao Hardcore KMeans (k = {k}) #####")
    hardcore_kmeans = KMeansClustering(k=k)
    avaliar_algoritmo(hardcore_kmeans, X, k, nome="Hardcore KMeans")
