"""Random Forest baseline (paper params). Run: python -m src.rf_model"""

import argparse

from sklearn.ensemble import RandomForestClassifier

from src.metrics import print_metrics
from src.preprocess import get_scenario

SCENARIOS = ['P7C', 'P7U', 'P6C', 'P6U']
PAPER_REF = {'P7C': 85.72, 'P7U': 85.65, 'P6C': 94.70, 'P6U': 94.69}


def run(data_path=None):
    for scenario in SCENARIOS:
        X_train, X_test, y_train, y_test = get_scenario(scenario, data_path)
        model = RandomForestClassifier(
            n_estimators=100, criterion='gini', random_state=0,
            min_samples_split=2, min_samples_leaf=1, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        print_metrics(y_test, model.predict(X_test), "Random Forest", scenario)
    print("\n========== Paper's reported RF accuracy ==========")
    for k, v in PAPER_REF.items():
        print(f"{k} : {v:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to Dataset_actual.csv")
    args = parser.parse_args()
    run(args.data)
