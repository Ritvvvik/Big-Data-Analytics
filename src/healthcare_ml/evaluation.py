from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(model, X_test, y_test) -> Dict[str, object]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    metrics["y_prob"] = y_prob
    metrics["y_pred"] = y_pred
    return metrics


def select_best_model(results: Dict[str, Dict[str, object]], primary_metric: str = "roc_auc") -> str:
    best_name = max(results.keys(), key=lambda k: results[k][primary_metric])
    return best_name
