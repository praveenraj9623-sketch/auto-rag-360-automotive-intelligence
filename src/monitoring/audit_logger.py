from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "artifacts" / "monitoring.db"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA journal_mode=WAL;")
    return connection


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_role TEXT NOT NULL,
                query TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                retrieved_count INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                confidence_score REAL NOT NULL,
                fallback_used INTEGER NOT NULL,
                access_denied_count INTEGER NOT NULL,
                answer_mode TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_summary TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL
            )
            """
        )


def log_rag_query(
    *,
    user_role: str,
    query: str,
    top_k: int,
    retrieved_count: int,
    latency_ms: float,
    confidence_score: float,
    fallback_used: bool,
    access_denied_count: int,
    answer_mode: str,
) -> None:
    initialize_database()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO rag_queries (
                timestamp,
                user_role,
                query,
                top_k,
                retrieved_count,
                latency_ms,
                confidence_score,
                fallback_used,
                access_denied_count,
                answer_mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _timestamp(),
                user_role,
                query,
                int(top_k),
                int(retrieved_count),
                float(latency_ms),
                float(confidence_score),
                int(bool(fallback_used)),
                int(access_denied_count),
                answer_mode,
            ),
        )


def log_ml_prediction(*, model_name: str, input_summary: Any, prediction: str, probability: float) -> None:
    initialize_database()
    if not isinstance(input_summary, str):
        input_summary = json.dumps(input_summary, sort_keys=True, default=str)
    input_summary = input_summary[:1000]
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO ml_predictions (
                timestamp,
                model_name,
                input_summary,
                prediction,
                probability
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (_timestamp(), model_name, input_summary, str(prediction), float(probability)),
        )


def safe_log_rag_query(**kwargs) -> None:
    try:
        log_rag_query(**kwargs)
    except Exception:
        return


def safe_log_ml_prediction(**kwargs) -> None:
    try:
        log_ml_prediction(**kwargs)
    except Exception:
        return
