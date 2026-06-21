from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
MODEL_PATH = MODEL_DIR / "maintenance_model.joblib"
METRICS_PATH = REPORT_DIR / "maintenance_metrics.json"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
TARGET_COLUMN = "Machine failure"


def ensure_ai4i_dataset(row_count: int = 500) -> None:
    required = [
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
        "Machine failure",
    ]
    if RAW_PATH.exists():
        try:
            df = pd.read_csv(RAW_PATH)
            canonical = _canonicalize_ai4i_columns(df)
            if len(canonical) > 0 and all(column in canonical.columns for column in required):
                return
        except Exception:
            pass

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    _generate_fallback_ai4i(row_count).to_csv(RAW_PATH, index=False)


def _generate_fallback_ai4i(row_count: int) -> pd.DataFrame:
    # Generated fallback data is synthetic AI4I-style data, not the real AI4I 2020 dataset.
    # It exists only so Phase 2 can run when the real CSV is absent.
    rng = random.Random(420)
    rows = []
    for index in range(row_count):
        machine_type = rng.choices(["L", "M", "H"], weights=[0.6, 0.3, 0.1], k=1)[0]
        air_temp = round(rng.normalvariate(300.0, 2.0), 1)
        process_temp = round(air_temp + rng.normalvariate(10.0, 1.0), 1)
        rotational_speed = int(max(1100, min(2900, rng.normalvariate(1530, 180))))
        torque = round(max(5, min(80, rng.normalvariate(40, 10))), 1)
        tool_wear = int(max(0, min(250, rng.normalvariate(110, 65))))
        heat_stress = process_temp > 312 and rotational_speed < 1450
        power_stress = torque > 55 and rotational_speed < 1400
        wear_stress = tool_wear > 200
        failure_probability = 0.02 + 0.18 * heat_stress + 0.26 * power_stress + 0.32 * wear_stress
        machine_failure = int(rng.random() < min(failure_probability, 0.92))
        rows.append(
            {
                "Type": machine_type,
                "Air temperature": air_temp,
                "Process temperature": process_temp,
                "Rotational speed": rotational_speed,
                "Torque": torque,
                "Tool wear": tool_wear,
                "Machine failure": machine_failure,
            }
        )
    return pd.DataFrame(rows)


def _canonicalize_ai4i_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Air temperature [K]": "Air temperature",
        "Process temperature [K]": "Process temperature",
        "Rotational speed [rpm]": "Rotational speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool wear",
    }
    cleaned = df.rename(columns=rename_map).copy()
    keep = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in keep if column not in cleaned.columns]
    if missing:
        raise ValueError(f"AI4I data is missing required columns: {missing}")
    cleaned = cleaned[keep].dropna()
    for column in FEATURE_COLUMNS[1:] + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    cleaned["Type"] = cleaned["Type"].astype(str).str.strip()
    return cleaned


def load_ai4i_data() -> pd.DataFrame:
    ensure_ai4i_dataset()
    return _canonicalize_ai4i_columns(pd.read_csv(RAW_PATH))


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
            ("numeric", StandardScaler(), FEATURE_COLUMNS[1:]),
        ]
    )


def _evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train_maintenance_model() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ai4i_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=360,
        stratify=y,
    )

    candidates = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=360),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=10,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=360,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    scored = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        scored[name] = {"model": model, "metrics": _evaluate_model(model, X_test, y_test)}

    best_name = max(scored, key=lambda key: scored[key]["metrics"]["f1"])
    best_model = scored[best_name]["model"]
    best_metrics = scored[best_name]["metrics"]

    artifact = {
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "model": best_model,
    }
    joblib.dump(artifact, MODEL_PATH)

    metrics_payload = {
        "selected_model": best_name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(np.mean(y)),
        **best_metrics,
        "candidate_metrics": {name: value["metrics"] for name, value in scored.items()},
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def main() -> None:
    metrics = train_maintenance_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
