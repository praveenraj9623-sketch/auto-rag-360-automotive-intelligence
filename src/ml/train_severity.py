from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_datasets import load_complaints


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
MODEL_PATH = MODEL_DIR / "severity_model.joblib"
METRICS_PATH = REPORT_DIR / "severity_metrics.json"

LABELS = ["High", "Low", "Medium"]

HIGH_KEYWORDS = [
    "crash",
    "fire",
    "injury",
    "injured",
    "death",
    "fatal",
    "brake failure",
    "steering loss",
    "service brakes",
    "reduced brake",
    "steering looseness",
]
MEDIUM_KEYWORDS = [
    "engine stall",
    "stall",
    "airbag warning",
    "air bag",
    "transmission issue",
    "transmission",
    "warning lamp",
]


def label_complaint_severity(text: str) -> str:
    normalized = text.lower()
    if any(keyword in normalized for keyword in HIGH_KEYWORDS):
        return "High"
    if any(keyword in normalized for keyword in MEDIUM_KEYWORDS):
        return "Medium"
    return "Low"


def load_labeled_complaints() -> pd.DataFrame:
    complaints = load_complaints().copy()
    crash_signal = complaints["CRASH"].fillna("N").astype(str).str.upper().map(lambda value: "crash" if value in {"Y", "1", "TRUE"} else "")
    fire_signal = complaints["FIRE"].fillna("N").astype(str).str.upper().map(lambda value: "fire" if value in {"Y", "1", "TRUE"} else "")
    injured_signal = pd.to_numeric(complaints.get("INJURED", 0), errors="coerce").fillna(0).map(lambda value: "injury" if value > 0 else "")
    deaths_signal = pd.to_numeric(complaints.get("DEATHS", 0), errors="coerce").fillna(0).map(lambda value: "death" if value > 0 else "")
    complaints["text"] = (
        complaints["CDESCR"].fillna("").astype(str)
        + " "
        + complaints["COMPONENT"].fillna("").astype(str)
        + " "
        + crash_signal
        + " "
        + fire_signal
        + " "
        + injured_signal
        + " "
        + deaths_signal
    )
    complaints["severity"] = complaints["text"].apply(label_complaint_severity)

    if complaints["severity"].nunique() < 2:
        raise ValueError("Complaint severity labeling produced fewer than two classes.")
    return complaints[["text", "severity", "COMPONENT"]]


def train_severity_model() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_labeled_complaints()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["severity"],
        test_size=0.2,
        random_state=360,
        stratify=df["severity"],
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=900)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=360)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(df["severity"].unique())
    metrics_payload = {
        "selected_model": "tfidf_logistic_regression",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_distribution": df["severity"].value_counts().to_dict(),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }

    joblib.dump(
        {
            "model_name": "tfidf_logistic_regression",
            "model": model,
            "labels": labels,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def main() -> None:
    metrics = train_severity_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
