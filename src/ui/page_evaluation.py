"""
AutoRAG 360 — RAG Evaluation
Radar chart, metric bars, latency histogram, hit/miss results.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.rag.evaluation import run_golden_question_evaluation
from src.ui.styles import PLOTLY_THEME, apply_plotly_theme, kpi_card, page_header, section_header


def _metrics_radar(report: dict) -> None:
    t = PLOTLY_THEME
    cats = ["Precision@3", "Precision@5", "Hit Rate", "MRR", "1−No-Ans"]
    vals = [
        report.get("precision_at_3", 0),
        report.get("precision_at_5", 0),
        report.get("hit_rate", 0),
        report.get("mean_reciprocal_rank", 0),
        1 - report.get("no_answer_rate", 0),
    ]
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=cats_c, fill="toself", name="RAG Performance",
        line=dict(color=t["accent"], width=2),
        fillcolor="rgba(31,111,235,0.12)",
        marker=dict(size=6, color=t["accent"]),
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.5] * (len(cats) + 1), theta=cats_c, name="Baseline 0.5",
        line=dict(color="#8b949e", width=1, dash="dot"),
        fillcolor="rgba(139,148,158,0.04)", fill="toself",
        marker=dict(size=0),
    ))
    fig = apply_plotly_theme(fig, "RAG Quality Radar", height=360)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(color="#6e7681", size=8), gridcolor="#21262d",
                            showline=False, tickvals=[0.25, 0.5, 0.75, 1.0]),
            angularaxis=dict(tickfont=dict(color="#8b949e", size=10), gridcolor="#21262d"),
            bgcolor="#161b22",
        ),
        legend=dict(font=dict(color="#8b949e", size=10), bgcolor="transparent"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _metric_bars(report: dict) -> None:
    t = PLOTLY_THEME
    metrics = [
        ("Precision@3", report.get("precision_at_3", 0)),
        ("Precision@5", report.get("precision_at_5", 0)),
        ("Hit Rate", report.get("hit_rate", 0)),
        ("MRR", report.get("mean_reciprocal_rank", 0)),
    ]
    labels, values = zip(*metrics)
    colors = [t["accent"], t["success"], t["warning"], "#8b949e"]

    fig = go.Figure(
        go.Bar(
            x=list(labels), y=list(values),
            marker=dict(color=colors, line=dict(width=0), opacity=0.85),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(color=t["font_color"], size=12),
            hovertemplate="<b>%{x}</b>: %{y:.3f}<extra></extra>",
        )
    )
    fig = apply_plotly_theme(fig, "Metric Scores", height=280)
    fig.update_layout(
        xaxis=dict(gridcolor="transparent"),
        yaxis=dict(range=[0, 1.2], gridcolor=t["grid_color"]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _latency_hist(details: list[dict]) -> None:
    t = PLOTLY_THEME
    latencies = [d.get("latency_ms", 0) for d in details if d.get("latency_ms")]
    if not latencies:
        return
    fig = go.Figure(
        go.Histogram(
            x=latencies, nbinsx=12,
            marker=dict(color=t["accent"], opacity=0.75, line=dict(color=t["bg"], width=1)),
            hovertemplate="Latency: %{x:.0f}ms<br>Count: %{y}<extra></extra>",
        )
    )
    fig = apply_plotly_theme(fig, "Latency Distribution (ms)", height=220)
    fig.update_layout(
        xaxis=dict(title="Latency (ms)", gridcolor=t["grid_color"]),
        yaxis=dict(title="Count", gridcolor=t["grid_color"]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _results_table(report: dict) -> None:
    for row in report.get("details", []):
        hit = row.get("hit", False)
        no_ans = row.get("no_answer", False)
        left_color = PLOTLY_THEME["success"] if hit else PLOTLY_THEME["danger"]
        latency = row.get("latency_ms", 0)
        rank = row.get("rank")
        sim = row.get("max_similarity", 0)

        st.markdown(
            f"""<div style="background:#161b22;border:1px solid #21262d;border-left:3px solid {left_color};
                border-radius:6px;padding:10px 14px;margin-bottom:6px">
                <div style="display:flex;align-items:flex-start;gap:10px">
                    <span style="font-size:0.875rem;margin-top:1px;flex-shrink:0">
                        {"✓" if hit else "✗"}
                    </span>
                    <div style="flex:1">
                        <div style="font-size:0.875rem;font-weight:500;color:#c9d1d9;margin-bottom:4px">
                            {row.get('question','')}
                        </div>
                        <div style="font-size:0.75rem;color:#6e7681;display:flex;flex-wrap:wrap;gap:10px">
                            <span>Role: <span style="color:#8b949e">{row.get('role','')}</span></span>
                            <span>Keyword: <span style="color:#58a6ff;font-family:monospace">{row.get('expected_keyword','')}</span></span>
                            <span>Rank: <span style="color:#8b949e">{rank if rank else '—'}</span></span>
                            <span>Similarity: <span style="color:#8b949e;font-family:monospace">{sim:.4f}</span></span>
                            <span>Latency: <span style="color:#8b949e">{latency:.0f}ms</span></span>
                            {'<span style="color:#f85149;font-weight:600">NO ANSWER</span>' if no_ans else ''}
                        </div>
                    </div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_evaluation_page() -> None:
    page_header(
        "RAG Evaluation · Golden Questions",
        "RAG Evaluation",
        "Golden-question retrieval evaluation with Precision@K, Hit Rate, MRR, "
        "latency, and no-answer rate. Fully role-aware.",
    )

    col_btn, _ = st.columns([2, 3])
    with col_btn:
        run_btn = st.button("Run Evaluation", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running evaluation across all golden questions..."):
            st.session_state["rag_eval_report"] = run_golden_question_evaluation()

    report = st.session_state.get("rag_eval_report")
    if not report:
        st.info("Click 'Run Evaluation' to execute the golden-question evaluation suite.")
        return

    # KPI row
    section_header("◆", "Evaluation Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, (label, value) in zip(
        [c1, c2, c3, c4, c5, c6],
        [
            ("Precision@3", f"{report['precision_at_3']:.3f}"),
            ("Precision@5", f"{report['precision_at_5']:.3f}"),
            ("Hit Rate", f"{report['hit_rate']:.3f}"),
            ("MRR", f"{report['mean_reciprocal_rank']:.3f}"),
            ("No-Answer", f"{report['no_answer_rate']:.3f}"),
            ("Avg Latency", f"{report['average_latency_ms']:.0f}ms"),
        ],
    ):
        with col:
            kpi_card(label, value)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    r_col, b_col = st.columns([2, 2])
    with r_col:
        _metrics_radar(report)
    with b_col:
        _metric_bars(report)
        _latency_hist(report.get("details", []))

    section_header("▦", "Golden Question Results")
    _results_table(report)
