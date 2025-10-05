from sklearn.datasets import load_iris, load_wine
from MLP_Classifier import run_mlp_on_dataset
from sklearn.datasets import load_iris, load_wine
from KnnHard import run_knn_hardcore
from KnnSkl import run_knn_sklearn
from MLP_Classifier import run_mlp_on_dataset

if __name__ == "__main__":
    # Rodar MLP
    run_mlp_on_dataset("Iris", load_iris().data, load_iris().target)
    run_mlp_on_dataset("Wine", load_wine().data, load_wine().target)

    # Rodar KNN Hardcore
    hc_results_iris = run_knn_hardcore("Iris", load_iris())
    hc_results_wine = run_knn_hardcore("Wine", load_wine())

    # Rodar KNN Sklearn (precisa ajustar igual hardcore)
    sk_results_iris = run_knn_sklearn("Iris", load_iris())
    sk_results_wine = run_knn_sklearn("Wine", load_wine())
