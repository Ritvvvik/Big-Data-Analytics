from __future__ import annotations

"""Model definitions with SMOTE balancing."""

from typing import Dict

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_models(preprocessor) -> Dict[str, Pipeline]:
    """Return model pipelines keyed by readable model names.

    Each pipeline uses:
    1) preprocessing
    2) SMOTE for class balancing
    3) classifier
    """
    smote = SMOTE(random_state=42)

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", smote),
                ("classifier", LogisticRegression(max_iter=1000, class_weight=None)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", smote),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
