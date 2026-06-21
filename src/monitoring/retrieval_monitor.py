from __future__ import annotations

import sqlite3

import pandas as pd

from src.monitoring.audit_logger import DB_PATH, initialize_database


def _read_sql(query: str, params: tuple = ()) -> pd.DataFrame:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(query, connection, params=params)


def get_recent_rag_queries(limit: int = 100, role: str | None = None) -> pd.DataFrame:
    if role and role != "All":
        return _read_sql(
            """
            SELECT timestamp, user_role, query, top_k, retrieved_count, latency_ms,
                   confidence_score, fallback_used, access_denied_count, answer_mode
            FROM rag_queries
            WHERE user_role = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (role, int(limit)),
        )
    return _read_sql(
        """
        SELECT timestamp, user_role, query, top_k, retrieved_count, latency_ms,
               confidence_score, fallback_used, access_denied_count, answer_mode
        FROM rag_queries
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (int(limit),),
    )


def get_recent_ml_predictions(limit: int = 100) -> pd.DataFrame:
    return _read_sql(
        """
        SELECT timestamp, model_name, input_summary, prediction, probability
        FROM ml_predictions
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (int(limit),),
    )


def queries_per_role() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT user_role, COUNT(*) AS query_count
        FROM rag_queries
        GROUP BY user_role
        ORDER BY query_count DESC
        """
    )


def average_latency_over_time() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT substr(timestamp, 1, 16) AS minute, AVG(latency_ms) AS average_latency_ms
        FROM rag_queries
        GROUP BY minute
        ORDER BY minute
        """
    )


def fallback_rate_over_time() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT substr(timestamp, 1, 16) AS minute, AVG(fallback_used) AS fallback_rate
        FROM rag_queries
        GROUP BY minute
        ORDER BY minute
        """
    )


def access_denied_by_role() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT user_role, SUM(access_denied_count) AS access_denied_count
        FROM rag_queries
        GROUP BY user_role
        ORDER BY access_denied_count DESC
        """
    )


def most_common_queries(limit: int = 10) -> pd.DataFrame:
    return _read_sql(
        """
        SELECT query, COUNT(*) AS query_count
        FROM rag_queries
        GROUP BY query
        ORDER BY query_count DESC
        LIMIT ?
        """,
        (int(limit),),
    )


def ml_predictions_per_model() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT model_name, COUNT(*) AS prediction_count
        FROM ml_predictions
        GROUP BY model_name
        ORDER BY prediction_count DESC
        """
    )


def latency_distribution() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT latency_ms
        FROM rag_queries
        ORDER BY timestamp DESC
        LIMIT 500
        """
    )
