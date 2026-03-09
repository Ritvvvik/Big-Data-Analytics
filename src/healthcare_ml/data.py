from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from zipfile import ZipFile

import pandas as pd
import requests


DATA_URL = "https://archive.ics.uci.edu/static/public/296/dataset_diabetes.zip"
RAW_CSV_NAME = "diabetic_data.csv"
IDS_CSV_NAME = "IDs_mapping.csv"


def download_dataset(data_dir: Path) -> Tuple[Path, Path]:
    """Download and extract the UCI Diabetes 130-US Hospitals dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "dataset_diabetes.zip"

    if not zip_path.exists():
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    with ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    diabetic_path = data_dir / RAW_CSV_NAME
    ids_path = data_dir / IDS_CSV_NAME
    return diabetic_path, ids_path


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load diabetic_data.csv into a DataFrame."""
    return pd.read_csv(csv_path)


def initial_exploration(df: pd.DataFrame) -> Dict[str, object]:
    """Return a compact EDA summary used in reporting."""
    summary = {
        "shape": df.shape,
        "column_count": len(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isna().sum().to_dict(),
        "target_distribution": (
            df["readmitted"].value_counts(normalize=True).to_dict()
            if "readmitted" in df.columns
            else {}
        ),
    }
    return summary
