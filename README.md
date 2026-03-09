# Healthcare ML Pipeline: Diabetes Readmission Prediction

A simple, faculty-friendly machine learning project using the UCI Diabetes 130-US Hospitals dataset.

## Libraries used (simple stack)

- pandas
- scikit-learn
- matplotlib
- seaborn
- streamlit (optional dashboard)

## 1) Get the dataset

Download from UCI:
- https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Place `diabetic_data.csv` at:
- `data/diabetic_data.csv`

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
4. Encode categorical features, scale numeric features
5. Train models:
   - Logistic Regression
   - Random Forest
6. Evaluate with:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC AUC
7. Choose best model by ROC AUC
8. Save model + metrics + plots

## Outputs

- `models/best_readmission_model.joblib`
- `artifacts/training_summary.json`
- `artifacts/sample_patient.csv`
- `figures/confusion_matrix.png`
- `figures/roc_curve.png`
- `figures/feature_importance.png` (if best model supports importance)

## Optional dashboard

```bash
streamlit run dashboard.py
```

The dashboard loads the saved model and predicts:
- readmission probability
- risk score (0–100)
- risk band (Low/Moderate/High)
