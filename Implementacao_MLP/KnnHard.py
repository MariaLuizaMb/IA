import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "Matrizes Hardcore"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def run_knn_hardcore(name, dataset, k_values=[1,3,5,7]):
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    for k in k_values:
        print(f"\n===== Avaliacao {name} (Hardcore) k = {k} =====")
        clf = KNearestNeighbors(k=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        precision = classification_report(y_test, y_pred, target_names=dataset.target_names, output_dict=True)['weighted avg']['precision']
        recall = classification_report(y_test, y_pred, target_names=dataset.target_names, output_dict=True)['weighted avg']['recall']
        print("Metricas KNN HardCore:\n")
        print(f"Acuracia: {acc:.2f}")
        print("Relatorio de Classificacao:")
        print(classification_report(y_test, y_pred, target_names=dataset.target_names))

        results.append({
            'dataset': name,
            'k': k,
            'accuracy': acc,
            'precision': precision,
            'recall': recall
        })
        
        # Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=dataset.target_names,
                    yticklabels=dataset.target_names)
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title(f"Matriz de Confusão {name} (Hardcore, k={k})")
        plt.savefig(os.path.join(RESULTS_DIR, f"knn_sklearn_confusion_matrix_{name}_k{k}.png"),
            dpi=300, bbox_inches="tight")
        
    return results