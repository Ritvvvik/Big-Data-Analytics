"""Command-line entrypoint for training."""

from pathlib import Path
import argparse

from src.healthcare_ml.pipeline import run_training_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train readmission prediction models")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/diabetic_data.csv",
        help="Path to diabetic_data.csv",
    )
    args = parser.parse_args()

    output = run_training_pipeline(base_dir=Path("."), csv_path=Path(args.csv))

    print("\nTraining complete")
    print("Best model:", output["best_model"])
    print("Saved model:", output["saved_model"])
