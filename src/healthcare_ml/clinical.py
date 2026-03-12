from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ClinicalPrediction:
    readmission_probability: float
    risk_score: int
    risk_band: str


def probability_to_risk_score(probability: float) -> int:
    return int(np.clip(np.round(probability * 100), 0, 100))


def classify_risk(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def predict_clinical_risk(model, patient_features: pd.DataFrame) -> ClinicalPrediction:
    prob = float(model.predict_proba(patient_features)[:, 1][0])
    score = probability_to_risk_score(prob)
    band = classify_risk(score)
    return ClinicalPrediction(prob, score, band)
