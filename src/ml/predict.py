from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.train_maintenance import FEATURE_COLUMNS as MAINTENANCE_FEATURES
from src.ml.train_maintenance import MODEL_PATH as MAINTENANCE_MODEL_PATH
from src.ml.train_maintenance import train_maintenance_model
from src.ml.train_severity import MODEL_PATH as SEVERITY_MODEL_PATH
from src.ml.train_severity import train_severity_model
from src.ml.train_vehicle_risk import FEATURE_COLUMNS as VEHICLE_FEATURES
from src.ml.train_vehicle_risk import MODEL_PATH as VEHICLE_MODEL_PATH
from src.ml.train_vehicle_risk import load_automobile_data, train_vehicle_risk_model
from src.monitoring.audit_logger import safe_log_ml_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

IMPORTANCE_PATHS = {
    "maintenance": REPORT_DIR / "maintenance_feature_importance.json",
    "severity": REPORT_DIR / "severity_feature_importance.json",
    "vehicle_risk": REPORT_DIR / "vehicle_risk_feature_importance.json",
}


def _load_importance(model_key: str) -> list[dict]:
    path = IMPORTANCE_PATHS[model_key]
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("top_features", [])[:10]


def _ensure_model(model_key: str) -> None:
    if model_key == "maintenance" and not MAINTENANCE_MODEL_PATH.exists():
        train_maintenance_model()
    elif model_key == "severity" and not SEVERITY_MODEL_PATH.exists():
        train_severity_model()
    elif model_key == "vehicle_risk" and not VEHICLE_MODEL_PATH.exists():
        train_vehicle_risk_model()


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def predict_maintenance_risk(input_dict: dict) -> dict:
    _ensure_model("maintenance")
    artifact = joblib.load(MAINTENANCE_MODEL_PATH)
    model = artifact["model"]

    row = {
        "Type": str(input_dict.get("Type", "M")),
        "Air temperature": _as_float(input_dict.get("Air temperature"), 300.0),
        "Process temperature": _as_float(input_dict.get("Process temperature"), 310.0),
        "Rotational speed": _as_float(input_dict.get("Rotational speed"), 1500.0),
        "Torque": _as_float(input_dict.get("Torque"), 40.0),
        "Tool wear": _as_float(input_dict.get("Tool wear"), 100.0),
    }
    X = pd.DataFrame([row], columns=MAINTENANCE_FEATURES)
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    failure_probability = float(probabilities[classes.index(1)] if 1 in classes else probabilities.max())
    prediction = int(failure_probability >= 0.5)
    confidence = float(max(failure_probability, 1.0 - failure_probability))
    if failure_probability >= 0.7:
        risk_level = "High"
    elif failure_probability >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    result = {
        "prediction": prediction,
        "failure_probability": failure_probability,
        "confidence": confidence,
        "risk_level": risk_level,
        "top_contributing_features": _load_importance("maintenance"),
    }
    safe_log_ml_prediction(
        model_name="maintenance",
        input_summary=row,
        prediction=result["risk_level"],
        probability=result["failure_probability"],
    )
    return result


def predict_complaint_severity(text: str) -> dict:
    _ensure_model("severity")
    artifact = joblib.load(SEVERITY_MODEL_PATH)
    model = artifact["model"]
    clean_text = text or ""
    probabilities = model.predict_proba([clean_text])[0]
    classes = list(model.classes_)
    best_index = int(np.argmax(probabilities))
    prediction = str(classes[best_index])
    result = {
        "prediction": prediction,
        "severity": prediction,
        "confidence": float(probabilities[best_index]),
        "class_probabilities": {str(label): float(probabilities[index]) for index, label in enumerate(classes)},
        "top_contributing_features": _load_importance("severity"),
    }
    safe_log_ml_prediction(
        model_name="complaint_severity",
        input_summary=clean_text[:300],
        prediction=result["severity"],
        probability=result["confidence"],
    )
    return result


def predict_vehicle_risk(input_dict: dict) -> dict:
    _ensure_model("vehicle_risk")
    artifact = joblib.load(VEHICLE_MODEL_PATH)
    model = artifact["model"]
    row = normalize_vehicle_input(input_dict)
    X = pd.DataFrame([row], columns=VEHICLE_FEATURES)
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    best_index = int(np.argmax(probabilities))
    prediction = str(classes[best_index])
    result = {
        "prediction": prediction,
        "risk_category": prediction,
        "confidence": float(probabilities[best_index]),
        "class_probabilities": {str(label): float(probabilities[index]) for index, label in enumerate(classes)},
        "top_contributing_features": _load_importance("vehicle_risk"),
    }
    safe_log_ml_prediction(
        model_name="vehicle_risk",
        input_summary=row,
        prediction=result["risk_category"],
        probability=result["confidence"],
    )
    return result


def normalize_vehicle_input(input_dict: dict) -> dict:
    return {
        "make": str(input_dict.get("make", "toyota")).strip().lower(),
        "body-style": str(input_dict.get("body-style", "sedan")).strip().lower(),
        "engine-type": str(input_dict.get("engine-type", "ohc")).strip().lower(),
        "horsepower": _as_float(input_dict.get("horsepower"), 110.0),
        "curb-weight": _as_float(input_dict.get("curb-weight"), 2500.0),
        "fuel-system": str(input_dict.get("fuel-system", "mpfi")).strip().lower(),
        "city-mpg": _as_float(input_dict.get("city-mpg"), 24.0),
        "highway-mpg": _as_float(input_dict.get("highway-mpg"), 30.0),
    }


def similar_vehicle_profiles(input_dict: dict, top_k: int = 5) -> pd.DataFrame:
    df = load_automobile_data()
    row = normalize_vehicle_input(input_dict)
    features = VEHICLE_FEATURES
    candidate = pd.concat([df[features], pd.DataFrame([row])], ignore_index=True)
    categorical = ["make", "body-style", "engine-type", "fuel-system"]
    numeric = ["horsepower", "curb-weight", "city-mpg", "highway-mpg"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    categorical_matrix = encoder.fit_transform(candidate[categorical])
    numeric_matrix = scaler.fit_transform(candidate[numeric])
    matrix = np.hstack([categorical_matrix, numeric_matrix])

    query_vector = matrix[-1:]
    candidate_vectors = matrix[:-1]
    similarities = cosine_similarity(query_vector, candidate_vectors).ravel()
    nearest_indices = np.argsort(similarities)[::-1][:top_k]
    result = df.iloc[nearest_indices][
        [
            "make",
            "body-style",
            "engine-type",
            "horsepower",
            "curb-weight",
            "city-mpg",
            "highway-mpg",
            "normalized-losses",
            "risk_category",
        ]
    ].copy()
    result.insert(0, "similarity", [round(float(similarities[index]), 3) for index in nearest_indices])
    return result
