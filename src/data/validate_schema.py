from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SchemaValidationResult:
    is_valid: bool
    missing_columns: list[str]


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> SchemaValidationResult:
    required = list(required_columns)
    missing = [column for column in required if column not in df.columns]
    return SchemaValidationResult(is_valid=not missing, missing_columns=missing)


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    result = validate_columns(df, required_columns)
    if not result.is_valid:
        missing = ", ".join(result.missing_columns)
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
