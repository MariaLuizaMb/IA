import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Carregar dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Dividir em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Valores de k
ks = [1, 3, 5, 7]

for k in ks:
    # Criar e treinar o modelo
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = knn.predict(X_test)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotar heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão (k={k})")
    
    
    # Salvar a imagem
    plt.savefig(f"matriz_confusao_k{k}.png")
    plt.close()  # Fecha a figura para não acumular memória
    
    # Métricas no console
    print(f"\n===== Resultados para k={k} =====")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Acuracia:", accuracy_score(y_test, y_pred))

print("\n Matrizes de confusão salvas como arquivos PNG.")
