from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Best Model)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_comparison(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for model_name, metrics in results.items():
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        auc = metrics["roc_auc"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top_feature_importance(model, feature_names, output_path: Path, top_n: int = 20) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], orient="h")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_shap_bar(feature_names, mean_abs_shap, output_path: Path, top_n: int = 20) -> None:
    idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=np.array(mean_abs_shap)[idx], y=np.array(feature_names)[idx], orient="h")
    plt.title(f"Top {top_n} SHAP Features")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
