import time
import psutil
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# importa sua implementação hardcore do arquivo separado
from Knn_HardCore.KnnHard import KNearestNeighbors   # ajuste o nome do arquivo .py para o seu

# ========================
# Função para medir memória
# ========================
def get_memory_usage_kb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 

# ========================
# Comparação
# ========================
def compare_knn(k_values=[1, 3, 5, 7]):
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for k in k_values:
        print(f"\n===== Comparacao para k={k} =====")

        # --- Hardcore ---
        hc = KNearestNeighbors(k=k)
        mem_before_hc = get_memory_usage_kb()
        start = time.perf_counter()
        hc.fit(X_train, y_train)
        y_pred_hc = hc.predict(X_test)
        end = time.perf_counter()
        mem_after_hc = get_memory_usage_kb()

        acc_hc = accuracy_score(y_test, y_pred_hc)
        prec_hc = precision_score(y_test, y_pred_hc, average="macro")
        rec_hc = recall_score(y_test, y_pred_hc, average="macro")

        print("\n[Hardcore]")
        print(f"Acuracia: {acc_hc:.4f} | Precisao: {prec_hc:.4f} | Revocacao: {rec_hc:.4f}")
        print(f"Tempo: {end-start:.6f} s | Memoria: {mem_after_hc-mem_before_hc:.3f} KB")
        print(classification_report(y_test, y_pred_hc, target_names=iris.target_names))

        # --- Sklearn ---
        sk = KNeighborsClassifier(n_neighbors=k)
        mem_before_sk = get_memory_usage_kb()
        start = time.perf_counter()
        sk.fit(X_train, y_train)
        y_pred_sk = sk.predict(X_test)
        end = time.perf_counter()
        mem_after_sk = get_memory_usage_kb()

        acc_sk = accuracy_score(y_test, y_pred_sk)
        prec_sk = precision_score(y_test, y_pred_sk, average="macro")
        rec_sk = recall_score(y_test, y_pred_sk, average="macro")

        print("\n[Sklearn]")
        print(f"Acuracia: {acc_sk:.4f} | Precisao: {prec_sk:.4f} | Revocacao: {rec_sk:.4f}")
        print(f"Tempo: {end-start:.6f} s | Memoria: {mem_after_sk-mem_before_sk:.3f} KB")
        print(classification_report(y_test, y_pred_sk, target_names=iris.target_names))


if __name__ == "__main__":
    compare_knn()
