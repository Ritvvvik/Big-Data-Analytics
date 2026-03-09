from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-class readmission target to binary: 1 for <30 days, else 0."""
    cleaned = df.copy()
    cleaned = cleaned[cleaned["readmitted"].notna()]
    cleaned["readmitted_binary"] = (cleaned["readmitted"] == "<30").astype(int)
    return cleaned


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    if {"num_medications", "num_lab_procedures", "num_procedures"}.issubset(
        engineered.columns
    ):
        engineered["care_intensity_index"] = (
            engineered["num_medications"]
            + engineered["num_lab_procedures"]
            + engineered["num_procedures"]
        )
    if {"number_outpatient", "number_emergency", "number_inpatient"}.issubset(
        engineered.columns
    ):
        engineered["total_prior_visits"] = (
            engineered["number_outpatient"]
            + engineered["number_emergency"]
            + engineered["number_inpatient"]
        )
    return engineered


def split_features_target(df: pd.DataFrame) -> SplitData:
    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "readmitted_binary",
    ]
    available_drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=available_drop_cols)
    y = df["readmitted_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return SplitData(X_train, X_test, y_train, y_test)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features
