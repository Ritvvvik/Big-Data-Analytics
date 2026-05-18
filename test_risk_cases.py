"""Test cases for Low, Medium, and High Risk patient predictions."""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.healthcare_ml.clinical import predict_clinical_risk


def create_low_risk_patient():
    """Create a LOW RISK patient profile."""
    return pd.DataFrame([{
        "race": "Caucasian",
        "gender": "Male",
        "age": "[30-40)",
        "time_in_hospital": 2,
        "num_lab_procedures": 20,
        "num_procedures": 1,
        "num_medications": 5,
        "number_outpatient": 1,
        "number_emergency": 0,
        "number_inpatient": 0,
        "A1Cresult": "Norm",
        "insulin": "No",
        "change": "No",
        "diabetesMed": "No",
        # Medications (Low Risk - minimal medications)
        "metformin": 0,
        "repaglinide": 0,
        "nateglinide": 0,
        "chlorpropamide": 0,
        "glimepiride": 0,
        "acetohexamide": 0,
        "glipizide": 0,
        "glyburide": 0,
        "tolbutamide": 0,
        "pioglitazone": 0,
        "rosiglitazone": 0,
        "acarbose": 0,
        "miglitol": 0,
        "troglitazone": 0,
        "tolazamide": 0,
        "examide": 0,
        "citoglipton": 0,
        "glyburide_metformin": 0,
        "glipizide_metformin": 0,
        "glimepiride_pioglitazone": 0,
        "metformin_rosiglitazone": 0,
        "metformin_pioglitazone": 0,
    }])


def create_medium_risk_patient():
    """Create a MEDIUM/MODERATE RISK patient profile."""
    return pd.DataFrame([{
        "race": "Caucasian",
        "gender": "Male",
        "age": "[50-60)",
        "time_in_hospital": 5,
        "num_lab_procedures": 35,
        "num_procedures": 3,
        "num_medications": 15,
        "number_outpatient": 3,
        "number_emergency": 2,
        "number_inpatient": 1,
        "A1Cresult": ">7",
        "insulin": "Steady",
        "change": "Ch",
        "diabetesMed": "Yes",
        # Medications (Medium Risk - moderate medications)
        "metformin": 1,
        "repaglinide": 0,
        "nateglinide": 0,
        "chlorpropamide": 0,
        "glimepiride": 1,
        "acetohexamide": 0,
        "glipizide": 1,
        "glyburide": 0,
        "tolbutamide": 0,
        "pioglitazone": 1,
        "rosiglitazone": 0,
        "acarbose": 0,
        "miglitol": 0,
        "troglitazone": 0,
        "tolazamide": 0,
        "examide": 0,
        "citoglipton": 0,
        "glyburide_metformin": 0,
        "glipizide_metformin": 0,
        "glimepiride_pioglitazone": 0,
        "metformin_rosiglitazone": 0,
        "metformin_pioglitazone": 0,
    }])


def create_high_risk_patient():
    """Create a HIGH RISK patient profile."""
    return pd.DataFrame([{
        "race": "AfricanAmerican",
        "gender": "Female",
        "age": "[70-80)",
        "time_in_hospital": 12,
        "num_lab_procedures": 58,
        "num_procedures": 8,
        "num_medications": 28,
        "number_outpatient": 8,
        "number_emergency": 6,
        "number_inpatient": 4,
        "A1Cresult": ">8",
        "insulin": "Up",
        "change": "Ch",
        "diabetesMed": "Yes",
        # Medications (High Risk - many medications)
        "metformin": 1,
        "repaglinide": 1,
        "nateglinide": 0,
        "chlorpropamide": 0,
        "glimepiride": 1,
        "acetohexamide": 0,
        "glipizide": 1,
        "glyburide": 1,
        "tolbutamide": 0,
        "pioglitazone": 1,
        "rosiglitazone": 1,
        "acarbose": 1,
        "miglitol": 0,
        "troglitazone": 0,
        "tolazamide": 0,
        "examide": 0,
        "citoglipton": 0,
        "glyburide_metformin": 1,
        "glipizide_metformin": 1,
        "glimepiride_pioglitazone": 0,
        "metformin_rosiglitazone": 1,
        "metformin_pioglitazone": 1,
    }])


def test_low_risk():
    """Test LOW RISK prediction."""
    model_path = Path("models/best_readmission_model.joblib")
    
    if not model_path.exists():
        print("❌ Model not found. Run: python run_pipeline.py")
        return
    
    model = joblib.load(model_path)
    patient = create_low_risk_patient()
    result = predict_clinical_risk(model, patient)
    
    print("\n" + "="*60)
    print("🟢 LOW RISK PATIENT TEST")
    print("="*60)
    print(f"Readmission Probability: {result.readmission_probability:.2%}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Risk Band: {result.risk_band}")
    print("="*60)
    
    assert result.risk_band == "Low", f"Expected 'Low', got '{result.risk_band}'"
    print("✅ LOW RISK TEST PASSED!\n")


def test_medium_risk():
    """Test MEDIUM/MODERATE RISK prediction."""
    model_path = Path("models/best_readmission_model.joblib")
    
    if not model_path.exists():
        print("❌ Model not found. Run: python run_pipeline.py")
        return
    
    model = joblib.load(model_path)
    patient = create_medium_risk_patient()
    result = predict_clinical_risk(model, patient)
    
    print("\n" + "="*60)
    print("🟡 MODERATE RISK PATIENT TEST")
    print("="*60)
    print(f"Readmission Probability: {result.readmission_probability:.2%}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Risk Band: {result.risk_band}")
    print("="*60)
    
    assert result.risk_band == "Moderate", f"Expected 'Moderate', got '{result.risk_band}'"
    print("✅ MODERATE RISK TEST PASSED!\n")


def test_high_risk():
    """Test HIGH RISK prediction."""
    model_path = Path("models/best_readmission_model.joblib")
    
    if not model_path.exists():
        print("❌ Model not found. Run: python run_pipeline.py")
        return
    
    model = joblib.load(model_path)
    patient = create_high_risk_patient()
    result = predict_clinical_risk(model, patient)
    
    print("\n" + "="*60)
    print("🔴 HIGH RISK PATIENT TEST")
    print("="*60)
    print(f"Readmission Probability: {result.readmission_probability:.2%}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Risk Band: {result.risk_band}")
    print("="*60)
    
    assert result.risk_band == "High", f"Expected 'High', got '{result.risk_band}'"
    print("✅ HIGH RISK TEST PASSED!\n")


def test_all_risk_cases():
    """Run all three risk case tests."""
    print("\n" + "="*60)
    print("RUNNING ALL RISK CASE TESTS")
    print("="*60)
    
    try:
        test_low_risk()
        test_medium_risk()
        test_high_risk()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")


if __name__ == "__main__":
    test_all_risk_cases()
