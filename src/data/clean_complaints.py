from __future__ import annotations

import pandas as pd

from src.data.validate_schema import require_columns


COMPLAINT_REQUIRED_COLUMNS = [
    "ODINO",
    "MFR_NAME",
    "COMPONENT",
    "VEHICLE_MODEL_YEAR",
    "MAKE",
    "MODEL",
    "CDESCR",
]

COMPLAINT_COLUMN_ALIASES = {
    "ODINO": "ODINO",
    "COMPLAINT ID": "ODINO",
    "MFR_NAME": "MFR_NAME",
    "MANUFACTURER": "MFR_NAME",
    "COMPONENT": "COMPONENT",
    "COMPONENTS": "COMPONENT",
    "VEHICLE_MODEL_YEAR": "VEHICLE_MODEL_YEAR",
    "MODEL YEAR": "VEHICLE_MODEL_YEAR",
    "MAKE": "MAKE",
    "MODEL": "MODEL",
    "CDESCR": "CDESCR",
    "DESCRIPTION": "CDESCR",
    "COMPLAINT DESCRIPTION": "CDESCR",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {}
    for column in df.columns:
        key = str(column).strip().upper()
        normalized[column] = COMPLAINT_COLUMN_ALIASES.get(key, column)
    return df.rename(columns=normalized)


def clean_complaints_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = _normalize_columns(df.copy())
    require_columns(cleaned, COMPLAINT_REQUIRED_COLUMNS, "complaints")

    for column in COMPLAINT_REQUIRED_COLUMNS:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    cleaned["VEHICLE_MODEL_YEAR"] = (
        pd.to_numeric(cleaned["VEHICLE_MODEL_YEAR"], errors="coerce").fillna(0).astype(int)
    )
    cleaned["document_id"] = cleaned.get("document_id", cleaned["ODINO"]).fillna(cleaned["ODINO"]).astype(str)
    cleaned["document_type"] = cleaned.get("document_type", "consumer_complaint")
    cleaned["sensitivity_level"] = cleaned.get("sensitivity_level", "public")
    cleaned["DATA_SOURCE"] = cleaned.get("DATA_SOURCE", "real_or_user_supplied_nhtsa_style")
    return cleaned
