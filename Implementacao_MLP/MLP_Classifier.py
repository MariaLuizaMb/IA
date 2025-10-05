import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay
)

RESULTS_DIR = "Matrizes MLP"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_mlp_on_dataset(name, X, y,
                       test_size=0.2, random_state=42,
                       hidden_layer_sizes=(64, 32), max_iter=1000):

    print("\n" + "="*60)
    print(f"Dataset: {name}")
    print("="*60)

    # 1) Divisão treino/teste (mantendo a proporção das classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Escalonamento das features (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3) Criação do modelo MLP e configuração
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state
    )

    # 4) Treino do modelo MLP 
    mlp.fit(X_train_scaled, y_train)

    # 5) Predição nos dados de teste
    y_pred = mlp.predict(X_test_scaled)

    # 6) Métricas de avaliação 
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Acuracia: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precisao (weighted): {prec:.4f}")
    print(f"Revocacao / Recall (weighted): {rec:.4f}")
    print("\nRelatorio de Classificacao:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Informações de convergência do MLP 
    try:
        print(f"Numero de iteracoes realizadas: {mlp.n_iter_}")
        if hasattr(mlp, "loss_curve_") and len(mlp.loss_curve_) > 0:
            print(f"Ultima perda registrada (loss): {mlp.loss_curve_[-1]:.6f}")
    except Exception:
        pass

    # 7) Matriz de confusão (plot e salvar imagem)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.ax_.set_title(f"Matriz de Confusão - {name}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")

    filename = os.path.join(RESULTS_DIR, f"mlp_matrizConfusao_{name}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[Imagem salva em: {filename}]")
    print("="*60 + "\n")

    plt.close()