from pathlib import Path

from src.healthcare_ml.pipeline import run_training_pipeline


if __name__ == "__main__":
    summary = run_training_pipeline(Path("."))
    print("Pipeline completed.")
    print("Best model:", summary["best_model"])
    for model_name, metrics in summary["model_metrics"].items():
        print(model_name, metrics)
