from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.healthcare_ml.clinical import predict_clinical_risk

st.set_page_config(page_title="Diabetes Readmission Risk Dashboard", layout="wide")
st.title("Diabetes Readmission Clinical Decision Dashboard")

model_path = Path("models/best_readmission_model.joblib")
sample_path = Path("artifacts/sample_patient.csv")

if not model_path.exists() or not sample_path.exists():
    st.warning("Model/sample data not found. Run `python run_pipeline.py` first.")
    st.stop()

model = joblib.load(model_path)
sample_df = pd.read_csv(sample_path)

st.subheader("Input Patient Features")
user_inputs = {}
for col in sample_df.columns:
    value = sample_df[col].iloc[0]
    if isinstance(value, (int, float)):
        user_inputs[col] = st.number_input(col, value=float(value))
    else:
        user_inputs[col] = st.text_input(col, value=str(value))

input_df = pd.DataFrame([user_inputs])

if st.button("Predict Readmission Risk"):
    prediction = predict_clinical_risk(model, input_df)
    st.metric("Readmission Probability", f"{prediction.readmission_probability:.2%}")
    st.metric("Risk Score (0-100)", prediction.risk_score)
    st.metric("Risk Band", prediction.risk_band)
