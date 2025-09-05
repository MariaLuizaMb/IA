import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# Função de distância Euclidiana
# ========================
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# ========================
# Classe KNN do zero
# ========================
class KNearestNeighbors:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            dist = euclidean_distance(x, self.X_train[i])
            distances.append([dist, self.y_train[i]])
        
        distances.sort(key=lambda x: x[0])  # ordena pelas distâncias
        neighbors = [label for (_, label) in distances[:self.k]]
        return Counter(neighbors).most_common(1)[0][0]
    
    def predict(self, X_test):
        return [self.predict_one(x) for x in X_test]

# ========================
# Carregar a base Iris
# ========================
iris = load_iris()
X = iris.data
y = iris.target

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# Avaliação para diferentes k
# ========================
for k in [1, 3, 5, 7]:
    print(f"\n===== Avaliacao para k = {k} =====")
    clf = KNearestNeighbors(k=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    print(f"Acuracia: {acc:.2f}")
    print("Relatorio de Classificacao:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Matriz de Confusão com acurácia embaixo
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão (k={k})")
    
    # adiciona a acurácia logo abaixo da matriz
    plt.figtext(0.5, -0.05, f"Acurácia: {acc:.2f}", ha="center", fontsize=12, fontweight="bold")
    
    plt.savefig(f"confusion_matrix_k{k}.png", dpi=300, bbox_inches="tight")
    plt.close()
