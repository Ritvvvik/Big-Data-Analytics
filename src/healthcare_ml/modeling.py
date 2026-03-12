from __future__ import annotations

"""Model definitions with SMOTE balancing."""

from typing import Dict

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except Exception:  # optional dependency
    XGBClassifier = None


def build_models(preprocessor) -> Dict[str, Pipeline]:
    """Return model pipelines keyed by readable model names.

    Each pipeline uses:
    1) preprocessing
    2) SMOTE for class balancing
    3) classifier
    """
    smote = SMOTE(random_state=42)

    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", smote),
                ("classifier", LogisticRegression(max_iter=1200)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", smote),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", smote),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return models
