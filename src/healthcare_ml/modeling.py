from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def build_model_pipelines(preprocessor) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        learning_rate=0.08,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=42,
                    ),
                ),
            ]
        )

    return models
