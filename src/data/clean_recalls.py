from __future__ import annotations

import pandas as pd

from src.data.validate_schema import require_columns


RECALL_REQUIRED_COLUMNS = [
    "CAMPNO",
    "MFGNAME",
    "COMPONENT",
    "YEARTXT",
    "MAKETXT",
    "MODELTXT",
    "SUMMARY",
    "CONSEQUENCE",
    "REMEDY",
]

RECALL_COLUMN_ALIASES = {
    "NHTSA CAMPAIGN NUMBER": "CAMPNO",
    "CAMPAIGN NUMBER": "CAMPNO",
    "CAMPNO": "CAMPNO",
    "MANUFACTURER": "MFGNAME",
    "MFGNAME": "MFGNAME",
    "COMPONENTS": "COMPONENT",
    "COMPONENT": "COMPONENT",
    "MODEL YEAR": "YEARTXT",
    "YEARTXT": "YEARTXT",
    "MAKE": "MAKETXT",
    "MAKETXT": "MAKETXT",
    "MODEL": "MODELTXT",
    "MODELTXT": "MODELTXT",
    "SUMMARY": "SUMMARY",
    "CONSEQUENCE": "CONSEQUENCE",
    "REMEDY": "REMEDY",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {}
    for column in df.columns:
        key = str(column).strip().upper()
        normalized[column] = RECALL_COLUMN_ALIASES.get(key, column)
    return df.rename(columns=normalized)


def clean_recalls_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = _normalize_columns(df.copy())
    require_columns(cleaned, RECALL_REQUIRED_COLUMNS, "recalls")

    for column in RECALL_REQUIRED_COLUMNS:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    cleaned["YEARTXT"] = pd.to_numeric(cleaned["YEARTXT"], errors="coerce").fillna(0).astype(int)
    cleaned["document_id"] = cleaned.get("document_id", cleaned["CAMPNO"]).fillna(cleaned["CAMPNO"]).astype(str)
    cleaned["document_type"] = cleaned.get("document_type", "recall_notice")
    cleaned["sensitivity_level"] = cleaned.get("sensitivity_level", "public")
    cleaned["DATA_SOURCE"] = cleaned.get("DATA_SOURCE", "real_or_user_supplied_nhtsa_style")
    return cleaned
