from __future__ import annotations

"""Preprocessing utilities for the readmission problem.

Design goal: keep this file easy to read in classroom/faculty review sessions.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "readmitted"
TARGET_BINARY = "readmitted_binary"


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup used by all models.

    - Converts '?' placeholders to missing values.
    - Drops duplicate rows.
    - Keeps sklearn-compatible missing values as np.nan.
    """
    cleaned = df.replace("?", np.nan).drop_duplicates().copy()
    return cleaned


def build_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary target: 1 for <30-day readmission, else 0."""
    out = df[df[TARGET_COLUMN].notna()].copy()
    out[TARGET_BINARY] = (out[TARGET_COLUMN] == "<30").astype(int)
    return out


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add two intuitive aggregate features frequently used in this dataset."""
    out = df.copy()

    if {"num_medications", "num_lab_procedures", "num_procedures"}.issubset(out.columns):
        out["care_intensity_index"] = (
            pd.to_numeric(out["num_medications"], errors="coerce").fillna(0)
            + pd.to_numeric(out["num_lab_procedures"], errors="coerce").fillna(0)
            + pd.to_numeric(out["num_procedures"], errors="coerce").fillna(0)
        )

    if {"number_outpatient", "number_emergency", "number_inpatient"}.issubset(out.columns):
        out["total_prior_visits"] = (
            pd.to_numeric(out["number_outpatient"], errors="coerce").fillna(0)
            + pd.to_numeric(out["number_emergency"], errors="coerce").fillna(0)
            + pd.to_numeric(out["number_inpatient"], errors="coerce").fillna(0)
        )

    return out


def train_test_data(df: pd.DataFrame, test_size: float = 0.2) -> DataSplit:
    """Build X/y and return a stratified train/test split."""
    ignore_cols = ["encounter_id", "patient_nbr", TARGET_COLUMN, TARGET_BINARY]
    ignore_cols = [c for c in ignore_cols if c in df.columns]

    X = df.drop(columns=ignore_cols)
    y = df[TARGET_BINARY]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )
    return DataSplit(X_train, X_test, y_train, y_test)


def make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create preprocessing pipelines for numeric and categorical columns."""
    X = X.copy()

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols
