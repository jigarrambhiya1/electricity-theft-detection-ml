"""XGBoost baseline (paper params). Run: python -m src.xgb_model"""

import argparse

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.metrics import print_metrics
from src.preprocess import get_scenario

SCENARIOS = ['P7C', 'P7U', 'P6C', 'P6U']
PAPER_REF = {'P7C': 85.67, 'P7U': 85.70, 'P6C': 91.21, 'P6U': 90.79}


def run(data_path=None):
    for scenario in SCENARIOS:
        X_train, X_test, y_train, y_test = get_scenario(scenario, data_path)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        n_classes = len(np.unique(y_train))
        model = XGBClassifier(
            max_depth=6, learning_rate=0.3, booster='gbtree',
            min_child_weight=1, scale_pos_weight=1,
            objective='multi:softprob', num_class=n_classes,
            eval_metric='mlogloss', n_jobs=-1, random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)
        print_metrics(y_test, model.predict(X_test), "XGBoost", scenario)
    print("\n========== Paper's reported XGB accuracy ==========")
    for k, v in PAPER_REF.items():
        print(f"{k} : {v:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to Dataset_actual.csv")
    args = parser.parse_args()
    run(args.data)
