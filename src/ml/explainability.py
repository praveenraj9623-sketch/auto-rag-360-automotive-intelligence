from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.ml.train_maintenance import FEATURE_COLUMNS as MAINTENANCE_FEATURES
from src.ml.train_maintenance import MODEL_PATH as MAINTENANCE_MODEL_PATH
from src.ml.train_maintenance import TARGET_COLUMN as MAINTENANCE_TARGET
from src.ml.train_maintenance import load_ai4i_data, train_maintenance_model
from src.ml.train_severity import MODEL_PATH as SEVERITY_MODEL_PATH
from src.ml.train_severity import load_labeled_complaints, train_severity_model
from src.ml.train_vehicle_risk import FEATURE_COLUMNS as VEHICLE_FEATURES
from src.ml.train_vehicle_risk import MODEL_PATH as VEHICLE_MODEL_PATH
from src.ml.train_vehicle_risk import TARGET_COLUMN as VEHICLE_TARGET
from src.ml.train_vehicle_risk import load_automobile_data, train_vehicle_risk_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
MAINTENANCE_IMPORTANCE_PATH = REPORT_DIR / "maintenance_feature_importance.json"
SEVERITY_IMPORTANCE_PATH = REPORT_DIR / "severity_feature_importance.json"
VEHICLE_IMPORTANCE_PATH = REPORT_DIR / "vehicle_risk_feature_importance.json"


def _top_importances(names: list[str], importances, stds, limit: int = 10) -> list[dict]:
    rows = []
    for name, importance, std in zip(names, importances, stds):
        rows.append(
            {
                "feature": str(name),
                "importance": float(importance),
                "std": float(std),
            }
        )
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows[:limit]


def _write_importance(path: Path, model_name: str, rows: list[dict]) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "method": "sklearn.inspection.permutation_importance", "top_features": rows}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return rows


def generate_maintenance_importance() -> list[dict]:
    if not MAINTENANCE_MODEL_PATH.exists():
        train_maintenance_model()
    artifact = joblib.load(MAINTENANCE_MODEL_PATH)
    model = artifact["model"]
    df = load_ai4i_data()
    X = df[MAINTENANCE_FEATURES]
    y = df[MAINTENANCE_TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=360, stratify=y)
    result = permutation_importance(model, X_test, y_test, scoring="f1", n_repeats=6, random_state=360, n_jobs=-1)
    rows = _top_importances(MAINTENANCE_FEATURES, result.importances_mean, result.importances_std)
    return _write_importance(MAINTENANCE_IMPORTANCE_PATH, "maintenance", rows)


def generate_severity_importance() -> list[dict]:
    if not SEVERITY_MODEL_PATH.exists():
        train_severity_model()
    artifact = joblib.load(SEVERITY_MODEL_PATH)
    pipeline = artifact["model"]
    df = load_labeled_complaints()
    X_train, X_test, _, y_test = train_test_split(
        df["text"],
        df["severity"],
        test_size=0.2,
        random_state=360,
        stratify=df["severity"],
    )
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    vectorizer.fit(X_train)
    X_test_vectorized = vectorizer.transform(X_test).toarray()
    classifier.fit(vectorizer.transform(X_train), df.loc[X_train.index, "severity"])
    result = permutation_importance(
        classifier,
        X_test_vectorized,
        y_test,
        scoring="f1_macro",
        n_repeats=4,
        random_state=360,
        n_jobs=-1,
    )
    feature_names = vectorizer.get_feature_names_out().tolist()
    rows = _top_importances(feature_names, result.importances_mean, result.importances_std)
    return _write_importance(SEVERITY_IMPORTANCE_PATH, "severity", rows)


def generate_vehicle_risk_importance() -> list[dict]:
    if not VEHICLE_MODEL_PATH.exists():
        train_vehicle_risk_model()
    artifact = joblib.load(VEHICLE_MODEL_PATH)
    model = artifact["model"]
    df = load_automobile_data()
    X = df[VEHICLE_FEATURES]
    y = df[VEHICLE_TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=360, stratify=y)
    result = permutation_importance(model, X_test, y_test, scoring="f1_macro", n_repeats=6, random_state=360, n_jobs=-1)
    rows = _top_importances(VEHICLE_FEATURES, result.importances_mean, result.importances_std)
    return _write_importance(VEHICLE_IMPORTANCE_PATH, "vehicle_risk", rows)


def generate_all_importances() -> dict[str, list[dict]]:
    return {
        "maintenance": generate_maintenance_importance(),
        "severity": generate_severity_importance(),
        "vehicle_risk": generate_vehicle_risk_importance(),
    }


def main() -> None:
    print(json.dumps(generate_all_importances(), indent=2))


if __name__ == "__main__":
    main()
