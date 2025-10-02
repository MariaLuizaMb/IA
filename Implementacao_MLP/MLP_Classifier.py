import numpy
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay
)

def run_mlp_on_dataset(name, X, y,
                       test_size=0.2, random_state=42,
                       hidden_layer_sizes=(64, 32), max_iter=1000):
    """
    Função que treina e avalia um MLPClassifier em um dataset.
    Exibe métricas e plota a matriz de confusão.
    """

    print("\n" + "="*60)
    print(f"Dataset: {name}")
    print("="*60)

    # 1) Divisão treino/teste (mantendo a proporção das classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Escalonamento das features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3) Criação do modelo MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state
    )

    # 4) Treino
    mlp.fit(X_train_scaled, y_train)

    # 5) Predição
    y_pred = mlp.predict(X_test_scaled)

    # 6) Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Acurácia: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precisão (weighted): {prec:.4f}")
    print(f"Revocação / Recall (weighted): {rec:.4f}")
    print("\nRelatório de Classificação:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Informações de convergência
    try:
        print(f"Número de iterações realizadas: {mlp.n_iter_}")
        if hasattr(mlp, "loss_curve_") and len(mlp.loss_curve_) > 0:
            print(f"Última perda registrada (loss): {mlp.loss_curve_[-1]:.6f}")
    except Exception:
        pass

    # 7) Matriz de confusão (plot)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.ax_.set_title(f"Matriz de Confusão - {name}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.show()


from sklearn.datasets import load_iris, load_wine

if __name__ == "__main__":
    # Carregar datasets
    iris = load_iris()
    wine = load_wine()

    # Executar modelo no dataset Iris
    run_mlp_on_dataset("Iris", iris.data, iris.target,
                       hidden_layer_sizes=(64, 32), max_iter=1000)

    # Executar modelo no dataset Wine
    run_mlp_on_dataset("Wine", wine.data, wine.target,
                       hidden_layer_sizes=(64, 32), max_iter=1000)