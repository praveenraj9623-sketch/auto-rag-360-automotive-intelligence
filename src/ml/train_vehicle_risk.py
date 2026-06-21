from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUTOMOBILE_PATH = RAW_DIR / "automobile.csv"
UCI_RAW_PATH = RAW_DIR / "automobile.csv.txt"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
MODEL_PATH = MODEL_DIR / "vehicle_risk_model.joblib"
METRICS_PATH = REPORT_DIR / "vehicle_risk_metrics.json"

FEATURE_COLUMNS = [
    "make",
    "body-style",
    "engine-type",
    "horsepower",
    "curb-weight",
    "fuel-system",
    "city-mpg",
    "highway-mpg",
]
TARGET_COLUMN = "risk_category"
AUTOMOBILE_REQUIRED_COLUMNS = FEATURE_COLUMNS + ["normalized-losses"]

UCI_COLUMNS = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]


def ensure_automobile_dataset(row_count: int = 150) -> None:
    if AUTOMOBILE_PATH.exists():
        try:
            df = pd.read_csv(AUTOMOBILE_PATH)
            if len(df) > 0 and all(column in df.columns for column in AUTOMOBILE_REQUIRED_COLUMNS):
                return
        except Exception:
            pass

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if UCI_RAW_PATH.exists():
        normalized = _load_uci_raw_automobile()
        if len(normalized) >= 50:
            normalized.to_csv(AUTOMOBILE_PATH, index=False)
            return

    _generate_fallback_automobile(row_count).to_csv(AUTOMOBILE_PATH, index=False)


def _load_uci_raw_automobile() -> pd.DataFrame:
    raw = pd.read_csv(UCI_RAW_PATH, header=None, names=UCI_COLUMNS, na_values="?")
    cleaned = raw[AUTOMOBILE_REQUIRED_COLUMNS].copy()
    for column in ["normalized-losses", "horsepower", "curb-weight", "city-mpg", "highway-mpg"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=AUTOMOBILE_REQUIRED_COLUMNS)
    for column in ["make", "body-style", "engine-type", "fuel-system"]:
        cleaned[column] = cleaned[column].astype(str).str.strip().str.lower()
    return cleaned.reset_index(drop=True)


def _generate_fallback_automobile(row_count: int) -> pd.DataFrame:
    # Generated fallback data is synthetic UCI Automobile-style data, not the real UCI dataset.
    # It exists only so Phase 2 can run when the real CSV is absent.
    rng = random.Random(421)
    makes = ["toyota", "honda", "mazda", "volvo", "bmw", "subaru", "nissan", "audi"]
    body_styles = ["sedan", "hatchback", "wagon", "hardtop", "convertible"]
    engine_types = ["ohc", "ohcv", "dohc", "l", "rotor"]
    fuel_systems = ["mpfi", "2bbl", "idi", "1bbl", "spdi"]
    rows = []
    for _ in range(row_count):
        horsepower = int(max(55, min(260, rng.normalvariate(115, 35))))
        curb_weight = int(max(1700, min(4200, rng.normalvariate(2550, 520))))
        city_mpg = int(max(12, min(49, rng.normalvariate(27, 7))))
        highway_mpg = int(max(city_mpg + 2, min(56, city_mpg + rng.randint(4, 10))))
        normalized_losses = int(
            max(
                60,
                min(
                    260,
                    85 + horsepower * 0.35 + (curb_weight - 2200) * 0.035 - city_mpg * 0.9 + rng.normalvariate(0, 18),
                ),
            )
        )
        rows.append(
            {
                "make": rng.choice(makes),
                "body-style": rng.choice(body_styles),
                "engine-type": rng.choice(engine_types),
                "horsepower": horsepower,
                "curb-weight": curb_weight,
                "fuel-system": rng.choice(fuel_systems),
                "city-mpg": city_mpg,
                "highway-mpg": highway_mpg,
                "normalized-losses": normalized_losses,
            }
        )
    return pd.DataFrame(rows)


def load_automobile_data() -> pd.DataFrame:
    ensure_automobile_dataset()
    df = pd.read_csv(AUTOMOBILE_PATH)
    missing = [column for column in AUTOMOBILE_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Automobile data is missing required columns: {missing}")

    cleaned = df[AUTOMOBILE_REQUIRED_COLUMNS].copy()
    for column in ["normalized-losses", "horsepower", "curb-weight", "city-mpg", "highway-mpg"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in ["make", "body-style", "engine-type", "fuel-system"]:
        cleaned[column] = cleaned[column].fillna("unknown").astype(str).str.strip().str.lower()
    cleaned = cleaned.dropna(subset=AUTOMOBILE_REQUIRED_COLUMNS).reset_index(drop=True)
    cleaned[TARGET_COLUMN] = pd.qcut(
        cleaned["normalized-losses"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    ).astype(str)
    return cleaned


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["make", "body-style", "engine-type", "fuel-system"]),
            ("numeric", StandardScaler(), ["horsepower", "curb-weight", "city-mpg", "highway-mpg"]),
        ]
    )


def train_vehicle_risk_model() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_automobile_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=360,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=8,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=360,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    labels = sorted(y.unique())

    metrics_payload = {
        "selected_model": "random_forest_classifier",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_distribution": y.value_counts().to_dict(),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }

    joblib.dump(
        {
            "model_name": "random_forest_classifier",
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "model": model,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def main() -> None:
    metrics = train_vehicle_risk_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
