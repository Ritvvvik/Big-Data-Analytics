"""Simple Streamlit dashboard for clinicians/faculty demos."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.healthcare_ml.clinical import predict_clinical_risk

st.set_page_config(page_title="Readmission Risk Dashboard", layout="wide")
st.title("Hospital Readmission Risk (Diabetes Dataset)")
st.caption("Input patient attributes and estimate 30-day readmission risk.")

model_path = Path("models/best_readmission_model.joblib")
sample_path = Path("artifacts/sample_patient.csv")

if not model_path.exists() or not sample_path.exists():
    st.warning("Please run `python run_pipeline.py` first to generate model/artifacts.")
    st.stop()

model = joblib.load(model_path)
default_row = pd.read_csv(sample_path).iloc[0]

st.subheader("Patient Input")
inputs = {}
for column_name, default_value in default_row.items():
    if pd.api.types.is_number(default_value):
        inputs[column_name] = st.number_input(column_name, value=float(default_value))
    else:
        inputs[column_name] = st.text_input(column_name, value=str(default_value))

if st.button("Predict"):
    patient_df = pd.DataFrame([inputs])
    result = predict_clinical_risk(model, patient_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Readmission Probability", f"{result.readmission_probability:.2%}")
    c2.metric("Risk Score (0-100)", result.risk_score)
    c3.metric("Risk Band", result.risk_band)
