from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from .data import download_dataset, initial_exploration, load_data
from .evaluation import evaluate_classifier, select_best_model
from .modeling import build_model_pipelines
from .preprocessing import (
    basic_feature_engineering,
    build_preprocessor,
    prepare_target,
    split_features_target,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_top_feature_importance,
)


def run_training_pipeline(base_dir: Path | str = ".") -> Dict[str, object]:
    base_dir = Path(base_dir)
    data_dir = base_dir / "data"
    model_dir = base_dir / "models"
    figure_dir = base_dir / "figures"
    artifact_dir = base_dir / "artifacts"

    diabetic_path, _ = download_dataset(data_dir)
    raw_df = load_data(diabetic_path)
    eda_summary = initial_exploration(raw_df)

    processed = prepare_target(raw_df)
    processed = basic_feature_engineering(processed)
    split = split_features_target(processed)

    preprocessor, _, _ = build_preprocessor(split.X_train)
    model_pipelines = build_model_pipelines(preprocessor)

    results: Dict[str, Dict[str, object]] = {}
    trained_models = {}

    for name, pipe in model_pipelines.items():
        pipe.fit(split.X_train, split.y_train)
        metrics = evaluate_classifier(pipe, split.X_test, split.y_test)
        results[name] = metrics
        trained_models[name] = pipe

    best_model_name = select_best_model(results, primary_metric="roc_auc")
    best_pipeline = trained_models[best_model_name]

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_readmission_model.joblib"
    joblib.dump(best_pipeline, model_path)

    summary = {
        "eda": eda_summary,
        "model_metrics": {
            k: {
                "accuracy": float(v["accuracy"]),
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1": float(v["f1"]),
                "roc_auc": float(v["roc_auc"]),
            }
            for k, v in results.items()
        },
        "best_model": best_model_name,
        "saved_model_path": str(model_path),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best_metrics = results[best_model_name]
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        figure_dir / "confusion_matrix.png",
        title=f"Confusion Matrix - {best_model_name}",
    )
    plot_roc_curve(
        best_metrics["roc_curve"]["fpr"],
        best_metrics["roc_curve"]["tpr"],
        best_metrics["roc_auc"],
        figure_dir / "roc_curve.png",
    )

    model_step = best_pipeline.named_steps["model"]
    preprocessor_step = best_pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor_step.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model_step.feature_importances_))]

    plot_top_feature_importance(
        model_step,
        feature_names,
        figure_dir / "feature_importance.png",
        top_n=20,
    )

    split.X_test.head(1).to_csv(artifact_dir / "sample_patient.csv", index=False)

    return summary
