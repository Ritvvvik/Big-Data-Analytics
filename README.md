# Healthcare Readmission ML Pipeline (UCI Diabetes 130-US Hospitals)

This project implements a modular, end-to-end machine learning pipeline to predict hospital readmission risk using the UCI `diabetic_data.csv` dataset.

## Architecture

1. **Data Ingestion Layer**: Downloads and loads dataset.
2. **Data Processing Layer**: Missing value handling, encoding, scaling, feature engineering, split.
3. **Machine Learning Layer**: Logistic Regression, Random Forest, XGBoost.
4. **Model Selection**: Picks best model using ROC AUC.
5. **Clinical Decision Output**: Returns probability + 0-100 risk score + risk band.
6. **Visualization Layer**: Feature importance, confusion matrix, ROC curve.
7. **Optional Dashboard**: Streamlit app for clinician-facing risk prediction.

## Project Structure

- `src/healthcare_ml/data.py` – data download, loading, EDA summary.
- `src/healthcare_ml/preprocessing.py` – target prep, feature engineering, preprocessing and split.
- `src/healthcare_ml/modeling.py` – model definitions.
- `src/healthcare_ml/evaluation.py` – metric calculation and model selection.
- `src/healthcare_ml/clinical.py` – clinical risk score utilities.
- `src/healthcare_ml/visualization.py` – chart generation.
- `src/healthcare_ml/pipeline.py` – orchestrates full training workflow.
- `run_pipeline.py` – training entrypoint.
- `dashboard.py` – Streamlit dashboard.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full training pipeline

```bash
python run_pipeline.py
```

Outputs:
- Trained model: `models/best_readmission_model.joblib`
- Summary metrics: `artifacts/training_summary.json`
- Visuals in `figures/`
- One sample patient row: `artifacts/sample_patient.csv`

## Run dashboard (optional)

```bash
streamlit run dashboard.py
```

## Notes

- The binary target is defined as readmitted within **<30 days** (`1`) vs all others (`0`).
- Missing values are handled with median/mode imputers.
- Categorical variables are one-hot encoded.
- Numeric variables are standardized.
