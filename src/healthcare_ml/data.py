from __future__ import annotations

"""Simple data ingestion utilities."""

from pathlib import Path
from typing import Dict

import pandas as pd


DEFAULT_DATA_PATH = Path("data/diabetic_data.csv")


def load_csv(csv_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the diabetes CSV from disk.

    This project keeps ingestion intentionally simple: place the file at
    ``data/diabetic_data.csv`` or pass a custom path.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download 'diabetic_data.csv' from UCI and place it in data/."
        )
    return pd.read_csv(csv_path)


def quick_eda(df: pd.DataFrame) -> Dict[str, object]:
    """Return a compact dataset overview for reporting."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values_per_column": df.isna().sum().to_dict(),
        "target_distribution": df["readmitted"].value_counts(normalize=True).to_dict(),
    }
