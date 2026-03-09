from __future__ import annotations

"""Main training pipeline (simple, classroom-friendly)."""

import json
from pathlib import Path
from typing import Dict

import joblib

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
from .visualization import plot_confusion_matrix, plot_roc_curve, plot_top_feature_importance


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
    }
    (artifacts_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

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

    classifier = best_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        plot_top_feature_importance(classifier, names, figures_dir / "feature_importance.png")

    split.X_test.head(1).to_csv(artifacts_dir / "sample_patient.csv", index=False)
    return summary
