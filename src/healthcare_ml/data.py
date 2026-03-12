from __future__ import annotations

"""Simple data ingestion utilities."""

from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd


DEFAULT_DATA_PATH = Path("data/diabetic_data.csv")
UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/296/diabetic_data.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/296/dataset_diabetes.zip"


def _download_dataset(target_csv: Path) -> None:
    """Download dataset from UCI if local CSV is missing."""
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Fast path if direct CSV is available
        urlretrieve(UCI_CSV_URL, target_csv)
        return
    except Exception:
        pass

    # Fallback: download zip and extract
    zip_path = target_csv.parent / "dataset_diabetes.zip"
    urlretrieve(UCI_ZIP_URL, zip_path)
    with ZipFile(zip_path, "r") as archive:
        archive.extract("diabetic_data.csv", target_csv.parent)


def load_csv(csv_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the diabetes CSV from disk.

    If the file is missing (e.g., dataset not stored in Git), the function tries
    to download it automatically from UCI.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        _download_dataset(csv_path)

    return pd.read_csv(csv_path)


def quick_eda(df: pd.DataFrame) -> Dict[str, object]:
    """Return a compact dataset overview for reporting."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values_per_column": df.isna().sum().to_dict(),
        "target_distribution": df["readmitted"].value_counts(normalize=True).to_dict(),
    }
