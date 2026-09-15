"""KNN baseline (n_neighbors=5) across P7C/P7U/P6C/P6U. Run: python -m src.knn_model"""

import argparse

from sklearn.neighbors import KNeighborsClassifier

from src.metrics import print_metrics
from src.preprocess import get_scenario

SCENARIOS = ['P7C', 'P7U', 'P6C', 'P6U']
PAPER_REF = {'P7C': 84.34, 'P7U': 84.10, 'P6C': 90.50, 'P6U': 89.65}


def run(data_path=None):
    for scenario in SCENARIOS:
        X_train, X_test, y_train, y_test = get_scenario(scenario, data_path)
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        model.fit(X_train, y_train)
        print_metrics(y_test, model.predict(X_test), "KNN", scenario)
    print("\n========== Paper's reported KNN accuracy ==========")
    for k, v in PAPER_REF.items():
        print(f"{k} : {v:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to Dataset_actual.csv")
    args = parser.parse_args()
    run(args.data)
