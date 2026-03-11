"""Clinical Readmission Risk Dashboard (Research Demo)"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.healthcare_ml.clinical import predict_clinical_risk

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Hospital Readmission Risk",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Readmission Risk Dashboard")
st.caption("Clinical Decision Support Prototype | Diabetes Dataset")

# ----------------------------------------------------
# LOAD MODEL + SAMPLE
# ----------------------------------------------------

model_path = Path("models/best_readmission_model.joblib")
sample_path = Path("artifacts/sample_patient.csv")

if not model_path.exists() or not sample_path.exists():
    st.warning("Run `python run_pipeline.py` first to generate the model.")
    st.stop()

model = joblib.load(model_path)
default_row = pd.read_csv(sample_path).iloc[0]

# ----------------------------------------------------
# PATIENT INPUT SECTION
# ----------------------------------------------------

st.subheader("Patient Information")

inputs = {}

col1, col2 = st.columns(2)

for i, (column_name, default_value) in enumerate(default_row.items()):

    if isinstance(default_value, (int, float)):
        widget = st.number_input(
            column_name,
            value=float(default_value)
        )
    else:
        widget = st.text_input(
            column_name,
            value=str(default_value)
        )

    inputs[column_name] = widget

# ----------------------------------------------------
# PREDICTION BUTTON
# ----------------------------------------------------

if st.button("Predict Readmission Risk"):

    patient_df = pd.DataFrame([inputs])
    result = predict_clinical_risk(model, patient_df)

    # ------------------------------------------------
    # RISK BAND COLOR
    # ------------------------------------------------

    if result.risk_band == "Low":
        band_color = "green"
    elif result.risk_band == "Medium":
        band_color = "orange"
    else:
        band_color = "red"

    # ------------------------------------------------
    # METRICS
    # ------------------------------------------------

    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Readmission Probability",
        f"{result.readmission_probability:.2%}"
    )

    c2.metric(
        "Risk Score (0–100)",
        result.risk_score
    )

    c3.markdown(
        f"<h2 style='color:{band_color}'>Risk Band: {result.risk_band}</h2>",
        unsafe_allow_html=True
    )

    # ------------------------------------------------
    # RISK VISUALIZATION
    # ------------------------------------------------

    st.subheader("Risk Visualization")

    probability_percent = int(result.readmission_probability * 100)

    st.progress(probability_percent)

    st.write(f"Estimated Risk: **{probability_percent}%**")

    # ------------------------------------------------
    # CLINICAL INTERPRETATION
    # ------------------------------------------------

    st.subheader("Clinical Interpretation")

    if result.risk_band == "Low":
        st.success(
            "Low likelihood of readmission. Standard follow-up recommended."
        )

    elif result.risk_band == "Medium":
        st.warning(
            "Moderate readmission risk. Consider monitoring and follow-up care."
        )

    else:
        st.error(
            "High risk of readmission. Recommend care coordination and discharge planning."
        )

# ----------------------------------------------------
# MODEL INFO
# ----------------------------------------------------

st.markdown("---")

st.subheader("Model Information")

st.write("""
**Dataset:** UCI Diabetes 130-US Hospitals Dataset (1999–2008)

**Prediction Task:** 30-day hospital readmission prediction

**Model:** Machine Learning classifier trained on patient encounter features.

**Use Case:** Clinical decision support for identifying high-risk patients.
""")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.caption("Research Prototype | Not for clinical use")
