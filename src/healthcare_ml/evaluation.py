from __future__ import annotations

"""Evaluation helpers."""

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test) -> Dict[str, object]:
    """Compute standard binary-classification metrics and ROC points."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
    }


def choose_best_model(results: Dict[str, Dict[str, object]], metric: str = "roc_auc") -> str:
    """Return the name of the model with the highest selected metric."""
    return max(results, key=lambda name: results[name][metric])
