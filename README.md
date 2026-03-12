# Healthcare ML Pipeline: Diabetes Readmission Prediction

A simple, faculty-friendly machine learning project using the UCI Diabetes 130-US Hospitals dataset.

## Application Architecture (Text Diagram)

> Binary images are removed to avoid platforms that do not support binary files in diffs/previews.

```mermaid
flowchart TD
    A[Data Sources<br>EHR / Labs / Demographics / Medications] --> B[Data Ingestion Layer<br>Load CSV (local or auto-download from UCI)]
    B --> C[Data Processing Layer<br>Cleaning + Missing Values + Encoding + Scaling + Feature Engineering]
    C --> D[Class Imbalance Handling<br>SMOTE]
    D --> E[Machine Learning Layer<br>Logistic Regression + Random Forest]
    E --> F[Evaluation Layer<br>Accuracy, Precision, Recall, F1, ROC AUC]
    F --> G[Model Selection<br>Best ROC AUC]
    G --> H[Clinical Decision Output<br>Readmission Probability + Risk Score + Risk Band]
    G --> I[Visualization Layer<br>Confusion Matrix + ROC + ROC Comparison + SHAP]
    H --> J[Streamlit Doctor Dashboard<br>Essential features only]
```
## Libraries used

- pandas
- scikit-learn
- imbalanced-learn (**SMOTE**)
- matplotlib
- seaborn
- shap (**SHAP explainability**)
- streamlit (optional dashboard)

## 1) Dataset handling

This repository does **not** store the dataset in Git.

- Preferred: place `diabetic_data.csv` at `data/diabetic_data.csv`
- If missing, the pipeline automatically downloads from UCI at runtime.

UCI source:
- https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## 2) Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --csv data/diabetic_data.csv
```

## 3) What the pipeline does

1. Load CSV with pandas
2. Clean data (`?` → missing values, remove duplicates)
3. Build binary target (`readmitted == "<30"`)
4. Encode categorical features + scale numeric features
5. Balance training data using **SMOTE**
6. Train models:
   - Logistic Regression
   - Random Forest
7. Evaluate with:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC AUC
8. Choose best model by ROC AUC
9. Save model + metrics + plots

## Outputs

- `models/best_readmission_model.joblib`
- `artifacts/training_summary.json`
- `artifacts/sample_patient.csv`
- `artifacts/dashboard_features.json` (limited features shown in Streamlit)
- `figures/confusion_matrix.png`
- `figures/roc_curve.png`
- `figures/roc_comparison.png`
- `figures/feature_importance.png` (if best model supports importance)
- `figures/shap_importance.png` (if SHAP available)

## Optional dashboard

```bash
streamlit run dashboard.py
```

The dashboard is doctor-friendly:
- Shows only essential features (instead of all columns)
- Uses dropdowns for clinical categorical fields
- Still fills other hidden model fields using default sample values

Predictions returned:
- readmission probability
- risk score (0–100)
- risk band (Low/Moderate/High)
