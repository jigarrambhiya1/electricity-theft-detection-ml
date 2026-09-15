"""Shared metric helper used by src model scripts."""

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def print_metrics(y_test, y_pred, model_name, scenario):
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    print(f"\n{'=' * 55}")
    print(f"  {model_name} | Scenario: {scenario}")
    print(f"{'=' * 55}")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n  Per-class Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
