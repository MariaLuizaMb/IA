import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

RESULTS_DIR = "Matrizes SKL"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_knn_sklearn(name, dataset, k_values=[1,3,5,7]):
    X, y = dataset.data, dataset.target

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    for k in k_values:
        # Criar e treinar o modelo
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Previsões
        y_pred = knn.predict(X_test)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        precision = classification_report(y_test, y_pred, target_names=dataset.target_names, output_dict=True)['weighted avg']['precision']
        recall = classification_report(y_test, y_pred, target_names=dataset.target_names, output_dict=True)['weighted avg']['recall']
        
        print(f"\n===== Avaliação {name} (Sklearn) k={k} =====")
        print(classification_report(y_test, y_pred, target_names=dataset.target_names))
        print(f"Acuracia: {acc:.2f}")

        results.append({
            'dataset': name,
            'k': k,
            'accuracy': acc,
            'precision': precision,
            'recall': recall
        })
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=dataset.target_names,
                    yticklabels=dataset.target_names)
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title(f"Matriz de Confusão {name} (Sklearn, k={k})")
        
        plt.savefig(os.path.join(RESULTS_DIR, f"knn_skl_matrizConfusao_{name}_k{k}.png"),
            dpi=300, bbox_inches="tight")
    
    return results