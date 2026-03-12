"""Simple Streamlit dashboard for clinicians/faculty demos."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.healthcare_ml.clinical import predict_clinical_risk

st.set_page_config(page_title="Readmission Risk Dashboard", layout="wide")
st.title("Hospital Readmission Risk (Diabetes Dataset)")
st.caption("Doctor-friendly form with key fields only.")

model_path = Path("models/best_readmission_model.joblib")
sample_path = Path("artifacts/sample_patient.csv")
dashboard_features_path = Path("artifacts/dashboard_features.json")

if not model_path.exists() or not sample_path.exists():
    st.warning("Please run `python run_pipeline.py` first to generate model/artifacts.")
    st.stop()

model = joblib.load(model_path)
default_row = pd.read_csv(sample_path).iloc[0]

if dashboard_features_path.exists():
    selected_features = json.loads(dashboard_features_path.read_text(encoding="utf-8"))
else:
    selected_features = list(default_row.index[:12])

selected_features = [f for f in selected_features if f in default_row.index]

st.subheader("Patient Input (Essential Features)")
inputs = default_row.to_dict()  # keep defaults for all unseen features


def _categorical_input(name: str, default_value: str):
    if name == "age":
        age_bands = [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
        ]
        default_idx = age_bands.index(default_value) if default_value in age_bands else 5
        return st.selectbox("Age Group", age_bands, index=default_idx)

    if name in {"gender", "change", "diabetesMed", "insulin", "A1Cresult", "race"}:
        options = [default_value]
        known = {
            "gender": ["Male", "Female", "Unknown/Invalid"],
            "change": ["No", "Ch"],
            "diabetesMed": ["No", "Yes"],
            "insulin": ["No", "Steady", "Up", "Down"],
            "A1Cresult": ["None", "Norm", ">7", ">8"],
            "race": ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
        }
        for v in known.get(name, []):
            if v not in options:
                options.append(v)
        return st.selectbox(name, options, index=0)

    return st.text_input(name, value=str(default_value))


for feature in selected_features:
    value = default_row[feature]
    if pd.api.types.is_number(value):
        inputs[feature] = st.number_input(feature, value=float(value), step=1.0)
    else:
        inputs[feature] = _categorical_input(feature, str(value))

if st.button("Predict"):
    patient_df = pd.DataFrame([inputs])
    result = predict_clinical_risk(model, patient_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Readmission Probability", f"{result.readmission_probability:.2%}")
    c2.metric("Risk Score (0-100)", result.risk_score)
    c3.metric("Risk Band", result.risk_band)
