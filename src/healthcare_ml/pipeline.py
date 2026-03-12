from __future__ import annotations

"""Main training pipeline (simple, classroom-friendly)."""

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from .data import load_csv, quick_eda
from .evaluation import choose_best_model, evaluate_model
from .modeling import build_models
from .preprocessing import (
    add_simple_features,
    build_binary_target,
    clean_raw_data,
    make_preprocessor,
    train_test_data,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_comparison,
    plot_roc_curve,
    plot_shap_bar,
    plot_top_feature_importance,
)


DOCTOR_INPUT_FEATURES: List[str] = [
    "race",
    "gender",
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]


def _save_shap_explanation(best_pipeline, X_test, figures_dir: Path) -> bool:
    """Create a simple SHAP bar plot for the best model if SHAP is available."""
    try:
        import shap
    except Exception:
        return False

    try:
        preprocessor = best_pipeline.named_steps["preprocessor"]
        classifier = best_pipeline.named_steps["classifier"]

        X_small = X_test.head(500)
        X_transformed = preprocessor.transform(X_small)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.Explainer(classifier, X_transformed)
        shap_values = explainer(X_transformed)
        values = shap_values.values
        if values.ndim == 3:  # multiclass shape safety
            values = values[:, :, 1]
        mean_abs_shap = np.abs(values).mean(axis=0)

        plot_shap_bar(feature_names, mean_abs_shap, figures_dir / "shap_importance.png", top_n=20)
        return True
    except Exception:
        return False


def run_training_pipeline(
    base_dir: Path | str = ".",
    csv_path: Path | str = "data/diabetic_data.csv",
) -> Dict[str, object]:
    """Run full ML workflow and save artifacts."""
    base_dir = Path(base_dir)
    models_dir = base_dir / "models"
    figures_dir = base_dir / "figures"
    artifacts_dir = base_dir / "artifacts"

    # 1) Load + inspect
    raw_df = load_csv(csv_path)
    eda = quick_eda(raw_df)

    # 2) Prepare data
    processed_df = clean_raw_data(raw_df)
    processed_df = build_binary_target(processed_df)
    processed_df = add_simple_features(processed_df)
    split = train_test_data(processed_df)
    preprocessor, _, _ = make_preprocessor(split.X_train)

    # 3) Train + evaluate
    models = build_models(preprocessor)
    results: Dict[str, Dict[str, object]] = {}
    trained = {}
    for model_name, pipeline in models.items():
        pipeline.fit(split.X_train, split.y_train)
        results[model_name] = evaluate_model(pipeline, split.X_test, split.y_test)
        trained[model_name] = pipeline

    # 4) Select best model by ROC AUC
    best_model_name = choose_best_model(results, metric="roc_auc")
    best_pipeline = trained[best_model_name]

    # 5) Save artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_readmission_model.joblib"
    joblib.dump(best_pipeline, model_path)

    doctor_features = [c for c in DOCTOR_INPUT_FEATURES if c in split.X_train.columns]
    (artifacts_dir / "dashboard_features.json").write_text(
        json.dumps(doctor_features, indent=2), encoding="utf-8"
    )

    summary = {
        "eda": eda,
        "model_metrics": {
            name: {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
            for name, metrics in results.items()
        },
        "best_model": best_model_name,
        "saved_model": str(model_path),
        "dashboard_features": doctor_features,
    }
    (artifacts_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # 6) Visualizations
    best_metrics = results[best_model_name]
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        figures_dir / "confusion_matrix.png",
        title=f"Confusion Matrix - {best_model_name}",
    )
    plot_roc_curve(
        best_metrics["roc_curve"]["fpr"],
        best_metrics["roc_curve"]["tpr"],
        best_metrics["roc_auc"],
        figures_dir / "roc_curve.png",
    )
    plot_roc_comparison(results, figures_dir / "roc_comparison.png")

    classifier = best_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        plot_top_feature_importance(classifier, names, figures_dir / "feature_importance.png")

    shap_created = _save_shap_explanation(best_pipeline, split.X_test, figures_dir)
    summary["shap_plot_created"] = shap_created
    (artifacts_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    split.X_test.head(1).to_csv(artifacts_dir / "sample_patient.csv", index=False)
    return summary
