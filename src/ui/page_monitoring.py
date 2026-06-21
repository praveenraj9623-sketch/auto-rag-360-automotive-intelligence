"""
AutoRAG 360 — Monitoring
Professional audit log dashboard with clean charts.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.monitoring.audit_logger import initialize_database
from src.monitoring.retrieval_monitor import (
    access_denied_by_role,
    average_latency_over_time,
    fallback_rate_over_time,
    get_recent_ml_predictions,
    get_recent_rag_queries,
    latency_distribution,
    ml_predictions_per_model,
    most_common_queries,
    queries_per_role,
)
from src.rag.access_control import get_roles
from src.ui.styles import PLOTLY_THEME, apply_plotly_theme, kpi_card, page_header, section_header


def _empty_state(message: str) -> None:
    st.markdown(
        f"""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
        padding:24px;text-align:center;color:#6e7681;font-size:0.875rem">
        {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _latency_area(df: pd.DataFrame) -> None:
    t = PLOTLY_THEME

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["average_latency_ms"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(31,111,235,0.08)",
            line=dict(color=t["accent"], width=2),
            name="Avg Latency (ms)",
            hovertemplate="Time: %{x}<br>Latency: %{y:.1f}ms<extra></extra>",
        )
    )

    fig = apply_plotly_theme(fig, "Average Latency Over Time", height=260)

    fig.update_layout(
        xaxis=dict(title="Time", gridcolor=t["grid_color"]),
        yaxis=dict(title="ms", gridcolor=t["grid_color"]),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _fallback_area(df: pd.DataFrame) -> None:
    t = PLOTLY_THEME

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["fallback_rate"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(210,153,34,0.08)",
            line=dict(color=t["warning"], width=2),
            name="Fallback Rate",
            hovertemplate="Time: %{x}<br>Rate: %{y:.1%}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(fig, "Fallback Rate Over Time", height=240)

    fig.update_layout(
        xaxis=dict(title="Time", gridcolor=t["grid_color"]),
        yaxis=dict(
            title="Fallback Rate",
            gridcolor=t["grid_color"],
            tickformat=".0%",
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _role_donut(df: pd.DataFrame) -> None:
    t = PLOTLY_THEME
    count_col = df.columns[1] if len(df.columns) > 1 else "query_count"

    fig = go.Figure(
        go.Pie(
            labels=df["user_role"],
            values=df[count_col],
            hole=0.55,
            marker=dict(
                colors=t["categorical"][: len(df)],
                line=dict(color=t["bg"], width=2),
            ),
            textinfo="percent+label",
            textfont=dict(size=11, family=t["font_family"]),
            hovertemplate="<b>%{label}</b><br>%{value} queries · %{percent}<extra></extra>",
        )
    )

    fig.add_annotation(
        text="By Role",
        x=0.5,
        y=0.5,
        font=dict(size=11, color="#8b949e"),
        showarrow=False,
    )

    fig = apply_plotly_theme(fig, "Query Distribution by Role", height=300)

    fig.update_layout(
        legend=dict(
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=80, t=48, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def _model_bar(df: pd.DataFrame) -> None:
    t = PLOTLY_THEME
    count_col = df.columns[1] if len(df.columns) > 1 else "prediction_count"

    fig = go.Figure(
        go.Bar(
            x=df["model_name"],
            y=df[count_col],
            marker=dict(
                color=t["categorical"][: len(df)],
                line=dict(width=0),
                opacity=0.85,
            ),
            text=df[count_col],
            textposition="outside",
            textfont=dict(color=t["font_color"], size=12),
            hovertemplate="<b>%{x}</b><br>Predictions: %{y}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(fig, "ML Predictions per Model", height=240)

    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=t["grid_color"]),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _access_bar(df: pd.DataFrame) -> None:
    t = PLOTLY_THEME
    count_col = df.columns[1] if len(df.columns) > 1 else "access_denied_count"

    fig = go.Figure(
        go.Bar(
            x=df["user_role"],
            y=df[count_col],
            marker=dict(
                color=t["danger"],
                line=dict(width=0),
                opacity=0.75,
            ),
            text=df[count_col],
            textposition="outside",
            textfont=dict(color=t["font_color"], size=12),
            hovertemplate="<b>%{x}</b><br>Denials: %{y}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(fig, "Access Denials by Role", height=260)

    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=t["grid_color"]),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _latency_box(df: pd.DataFrame) -> None:
    t = PLOTLY_THEME

    fig = go.Figure(
        go.Box(
            y=df["latency_ms"],
            name="Latency",
            marker=dict(color=t["accent"], size=4),
            line=dict(color=t["accent"], width=1.5),
            fillcolor="rgba(31,111,235,0.1)",
            boxmean="sd",
            hovertemplate="Latency: %{y:.0f}ms<extra></extra>",
        )
    )

    fig = apply_plotly_theme(fig, "Latency Distribution", height=240)

    fig.update_layout(
        yaxis=dict(title="ms", gridcolor=t["grid_color"]),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_monitoring_page() -> None:
    initialize_database()

    page_header(
        "Monitoring · SQLite Audit Logs",
        "System Monitoring",
        "Real-time audit logs for RAG queries, ML predictions, latency, "
        "fallback rates, and access control violations.",
    )

    q_tab, pred_tab, lat_tab, fb_tab, acc_tab = st.tabs(
        ["Query Logs", "ML Predictions", "Latency", "Fallbacks", "Access Denials"]
    )

    with q_tab:
        role_filter = st.selectbox(
            "Role filter",
            ["All", *get_roles()],
            key="mon_role",
        )

        role_counts = queries_per_role()

        c1, c2 = st.columns([1, 1])

        with c1:
            if not role_counts.empty:
                _role_donut(role_counts)
            else:
                _empty_state("No query role data yet. Use the RAG Assistant to generate logs.")

        with c2:
            section_header("▦", "Most Common Queries")
            common = most_common_queries()

            if common.empty:
                _empty_state("No common-query data yet.")
            else:
                st.dataframe(common, use_container_width=True, hide_index=True)

        section_header("◻", "Recent Query Logs")
        recent = get_recent_rag_queries(limit=50, role=role_filter)

        if recent.empty:
            _empty_state("No query logs yet.")
        else:
            st.dataframe(recent, use_container_width=True, hide_index=True)

    with pred_tab:
        model_counts = ml_predictions_per_model()

        if not model_counts.empty:
            _model_bar(model_counts)
        else:
            _empty_state(
                "No ML prediction logs yet. Use Predictive Maintenance or Vehicle Risk pages."
            )

        section_header("◻", "Recent Prediction Logs")
        recent_pred = get_recent_ml_predictions(limit=50)

        if recent_pred.empty:
            _empty_state("No prediction logs yet.")
        else:
            st.dataframe(recent_pred, use_container_width=True, hide_index=True)

    with lat_tab:
        lat_df = average_latency_over_time()

        if lat_df.empty:
            _empty_state("No latency data yet.")
        else:
            _latency_area(lat_df)

        dist_df = latency_distribution()

        if not dist_df.empty:
            box_col, stat_col = st.columns([2, 1])

            with box_col:
                _latency_box(dist_df)

            with stat_col:
                p50 = dist_df["latency_ms"].quantile(0.50)
                p95 = dist_df["latency_ms"].quantile(0.95)
                p99 = dist_df["latency_ms"].quantile(0.99)

                st.markdown("<br>", unsafe_allow_html=True)

                for label, value, color in [
                    ("P50 Latency", f"{p50:.0f}ms", "#3fb950"),
                    ("P95 Latency", f"{p95:.0f}ms", "#d29922"),
                    ("P99 Latency", f"{p99:.0f}ms", "#f85149"),
                ]:
                    kpi_card(label, value, value_color=color)
                    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    with fb_tab:
        fb_df = fallback_rate_over_time()

        if fb_df.empty:
            _empty_state(
                "No fallback data yet. Fallbacks are logged when no LLM key is configured."
            )
        else:
            _fallback_area(fb_df)

    with acc_tab:
        acc_df = access_denied_by_role()

        if acc_df.empty:
            _empty_state(
                "No access-denial data yet. Use the Access Control Demo to generate entries."
            )
        else:
            ac1, ac2 = st.columns([3, 1])

            with ac1:
                _access_bar(acc_df)

            with ac2:
                count_col = (
                    acc_df.columns[1]
                    if len(acc_df.columns) > 1
                    else "access_denied_count"
                )

                total = int(acc_df[count_col].sum())

                st.markdown("<br>", unsafe_allow_html=True)

                kpi_card(
                    "Total Denials",
                    f"{total:,}",
                    "Role-based access control is working",
                    value_color="#f85149",
                )