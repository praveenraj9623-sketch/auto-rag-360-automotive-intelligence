"""
AutoRAG 360 — Executive Overview
Professional KPI dashboard with clean Plotly charts.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.styles import PLOTLY_THEME, apply_plotly_theme, kpi_card, page_header, section_header


def render_data_source_badge(data_source_label: str) -> None:
    is_real = "real" in data_source_label.lower()
    style = "green" if is_real else "yellow"
    st.markdown(
        f'<span class="badge badge-{style}">{data_source_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def _kpi_row(recalls, complaints, maintenance_health, prediction_count) -> None:
    top_components = recalls["COMPONENT"].value_counts()
    top_component = top_components.index[0] if not top_components.empty else "N/A"
    f1 = maintenance_health.get("f1", 0) if maintenance_health else None
    roc = maintenance_health.get("roc_auc", 0) if maintenance_health else None

    cols = st.columns(6)
    data = [
        ("Total Recalls", f"{len(recalls):,}", "Indexed in corpus"),
        ("Total Complaints", f"{len(complaints):,}", "Indexed in corpus"),
        ("Predictions Made", f"{prediction_count:,}", "Since session start"),
        ("Top Component", top_component, "By recall volume"),
        ("Model F1", f"{f1:.3f}" if f1 is not None else "—", "Maintenance model"),
        ("ROC-AUC", f"{roc:.3f}" if roc is not None else "—", "Maintenance model"),
    ]

    for col, (label, value, sub) in zip(cols, data):
        with col:
            kpi_card(label, value, sub)


def _component_bar(recalls: pd.DataFrame) -> None:
    top = recalls["COMPONENT"].value_counts().head(10).reset_index()
    top.columns = ["component", "count"]
    t = PLOTLY_THEME

    fig = go.Figure(
        go.Bar(
            x=top["count"],
            y=top["component"],
            orientation="h",
            marker=dict(color=t["accent"], line=dict(width=0), opacity=0.85),
            text=top["count"],
            textposition="outside",
            textfont=dict(color=t["font_color"], size=11),
            hovertemplate="<b>%{y}</b><br>Recalls: %{x}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(fig, "Top 10 Components by Recall Count", height=360)

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending",
            showgrid=False,
            tickfont=dict(size=11),
        ),
        xaxis=dict(
            gridcolor=t["grid_color"],
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _component_treemap(recalls: pd.DataFrame) -> None:
    import plotly.express as px

    counts = recalls["COMPONENT"].value_counts().reset_index()
    counts.columns = ["component", "count"]

    fig = px.treemap(
        counts.head(16),
        path=["component"],
        values="count",
        color="count",
        color_continuous_scale=[[0, "#21262d"], [1.0, "#1f6feb"]],
    )

    fig.update_traces(
        textinfo="label+value",
        textfont=dict(size=12, family="Inter"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    )

    fig = apply_plotly_theme(fig, "Component Distribution", height=360)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=4, r=4, t=40, b=4),
    )

    st.plotly_chart(fig, use_container_width=True)


def _model_health_gauges(maintenance_health: dict) -> None:
    f1 = maintenance_health.get("f1", 0)
    roc = maintenance_health.get("roc_auc", 0)
    t = PLOTLY_THEME

    def _gauge_color(value):
        if value >= 0.8:
            return t["success"]
        if value >= 0.6:
            return t["warning"]
        return t["danger"]

    fig = go.Figure()

    for i, (label, value) in enumerate([("F1 Score", f1), ("ROC-AUC", roc)]):
        color = _gauge_color(value)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={
                    "text": label,
                    "font": {
                        "size": 13,
                        "color": t["font_color"],
                        "family": t["font_family"],
                    },
                },
                number={
                    "font": {
                        "color": color,
                        "size": 28,
                        "family": t["font_family"],
                    },
                    "valueformat": ".3f",
                },
                gauge={
                    "axis": {
                        "range": [0, 1],
                        "tickcolor": t["grid_color"],
                        "tickfont": {
                            "size": 10,
                            "color": t.get("text_secondary", "#8b949e"),
                        },
                    },
                    "bar": {"color": color, "thickness": 0.25},
                    "bgcolor": t["paper_bg"],
                    "borderwidth": 1,
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0, 0.6], "color": "rgba(248,81,73,0.08)"},
                        {"range": [0.6, 0.8], "color": "rgba(210,153,34,0.08)"},
                        {"range": [0.8, 1.0], "color": "rgba(63,185,80,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 2},
                        "thickness": 0.8,
                        "value": value,
                    },
                },
                domain={"x": [i * 0.5, (i + 1) * 0.5 - 0.02], "y": [0, 1]},
            )
        )

    fig = apply_plotly_theme(fig, "Maintenance Model Health", height=260)
    fig.update_layout(margin=dict(l=16, r=16, t=48, b=16))

    st.plotly_chart(fig, use_container_width=True)


def _complaint_donut(complaints: pd.DataFrame) -> None:
    counts = complaints["COMPONENT"].value_counts().head(8)
    remainder = complaints["COMPONENT"].value_counts().iloc[8:].sum()

    if remainder > 0:
        counts["Other"] = remainder

    t = PLOTLY_THEME

    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.55,
            marker=dict(
                colors=t["categorical"],
                line=dict(color=t["bg"], width=2),
            ),
            textinfo="percent",
            textfont=dict(size=11, family=t["font_family"]),
            hovertemplate="<b>%{label}</b><br>%{value} complaints<br>%{percent}<extra></extra>",
        )
    )

    fig.add_annotation(
        text="Complaint<br>Distribution",
        x=0.5,
        y=0.5,
        font=dict(size=10, color="#8b949e", family="Inter"),
        showarrow=False,
        align="center",
    )

    fig = apply_plotly_theme(fig, "Complaints by Component", height=320)

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.0,
            y=0.5,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=100, t=48, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_overview(
    recalls,
    complaints,
    data_source_label: str,
    maintenance_health: dict | None = None,
    prediction_count: int = 0,
) -> None:
    page_header(
        "Executive Overview",
        "AutoRAG 360 Dashboard",
        "Recall intelligence, complaint analytics, ML model health, and operational KPIs.",
    )

    render_data_source_badge(data_source_label)

    _kpi_row(recalls, complaints, maintenance_health, prediction_count)
    st.markdown("<br>", unsafe_allow_html=True)

    section_header("📊", "Recall Analytics")
    c1, c2 = st.columns([3, 2])

    with c1:
        _component_treemap(recalls)

    with c2:
        _component_bar(recalls)

    section_header("🧠", "Model Health & Complaint Distribution")
    c3, c4 = st.columns([1, 1])

    with c3:
        if maintenance_health:
            _model_health_gauges(maintenance_health)
        else:
            st.info("Train the maintenance model to see F1 and ROC-AUC gauges.")

    with c4:
        _complaint_donut(complaints)