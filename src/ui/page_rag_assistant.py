"""
AutoRAG 360 — RAG Assistant
Professional chat interface with confidence gauge and citation table.
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.monitoring.audit_logger import safe_log_rag_query
from src.rag.access_control import get_roles
from src.rag.generator import generate_grounded_answer
from src.ui.page_overview import render_data_source_badge
from src.ui.styles import PLOTLY_THEME, apply_plotly_theme, page_header, section_header


def _sources_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        meta = result.get("metadata", {})
        rows.append({
            "Source": meta.get("source", "unknown"),
            "Document Type": meta.get("document_type", "unknown"),
            "Sensitivity": meta.get("sensitivity_level", "unknown"),
            "Score": round(float(result.get("score", 0.0)), 4),
        })
    return pd.DataFrame(rows)


def _confidence_gauge(score: float) -> None:
    t = PLOTLY_THEME
    color = t["success"] if score > 0.6 else t["warning"] if score > 0.3 else t["danger"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"color": color, "size": 28, "family": t["font_family"]},
                    "valueformat": ".1f"},
            title={"text": "Retrieval Confidence",
                   "font": {"size": 12, "color": "#8b949e", "family": t["font_family"]}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 9, "color": "#6e7681"}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": t["paper_bg"],
                "bordercolor": "#30363d",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, 33], "color": "rgba(248,81,73,0.08)"},
                    {"range": [33, 66], "color": "rgba(210,153,34,0.06)"},
                    {"range": [66, 100], "color": "rgba(63,185,80,0.08)"},
                ],
                "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.8, "value": score * 100},
            },
        )
    )
    fig = apply_plotly_theme(fig, "", height=200)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _answer_mode_html(mode: str) -> str:
    styles = {
        "extractive": ("yellow", "Extractive"),
        "groq": ("green", "Groq LLM"),
        "gemini": ("blue", "Gemini LLM"),
    }
    style, label = styles.get(mode, ("gray", mode.capitalize()))
    return f'<span class="badge badge-{style}">{label}</span>'


def _render_sources(results: list[dict]) -> None:
    doc_icons = {
        "recall": "⬜", "complaint": "△", "service_manual": "◆",
        "quality_report": "◈", "recall_policy": "◉", "supplier_notice": "▣",
    }
    sens_colors = {"high": "#f85149", "medium": "#d29922", "low": "#3fb950"}

    for result in results:
        meta = result.get("metadata", {})
        doc_type = meta.get("document_type", "unknown")
        sensitivity = meta.get("sensitivity_level", "unknown")
        source = meta.get("source", "unknown")
        score = float(result.get("score", 0.0))
        icon = doc_icons.get(doc_type, "◻")
        s_color = sens_colors.get(sensitivity.lower(), "#6e7681")
        score_color = PLOTLY_THEME["success"] if score > 0.5 else PLOTLY_THEME["warning"] if score > 0.2 else "#6e7681"

        st.markdown(
            f"""<div class="source-item">
                <span style="font-size:1rem;opacity:0.6">{icon}</span>
                <div style="flex:1;min-width:0">
                    <div style="font-weight:600;color:#e6edf3;font-size:0.8125rem;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{source}</div>
                    <div style="color:#6e7681;font-size:0.75rem;margin-top:2px">
                        {doc_type.replace('_',' ').title()}
                        &nbsp;·&nbsp;
                        <span style="color:{s_color}">{sensitivity.upper()}</span>
                    </div>
                </div>
                <span style="font-size:0.75rem;font-weight:600;color:{score_color};
                    font-family:monospace;white-space:nowrap">{score:.4f}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def render_rag_assistant(retriever, data_source_label: str) -> None:
    page_header(
        "RAG Assistant",
        "Automotive RAG Assistant",
        "Query recalls, complaints, service manuals, and quality documents. "
        "Answers are grounded in retrieved sources with role-based access control.",
    )
    render_data_source_badge(data_source_label)

    # Controls
    col_q, col_r = st.columns([4, 1])
    with col_r:
        role = st.selectbox("Role", get_roles(), index=0)
    with col_q:
        query = st.text_input(
            "Query",
            value="What brake remedies are available?",
            placeholder="Ask about recalls, complaints, or technical documents...",
        )

    if not query.strip():
        st.markdown(
            """<div style="text-align:center;padding:48px 20px;color:#6e7681">
                <div style="font-size:0.875rem">Enter a query above to retrieve automotive documents</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Retrieving documents..."):
        start = time.perf_counter()
        retrieval = retriever.retrieve(query=query, role=role, top_k=5)
        latency_ms = (time.perf_counter() - start) * 1000

    with st.spinner("Generating answer..."):
        answer = generate_grounded_answer(query, retrieval["results"])

    confidence_score = max(
        (float(r.get("score", 0.0)) for r in retrieval["results"]), default=0.0
    )
    preview = retriever.preview_access(query=query, roles=[role], top_k=10)
    access_denied_count = sum(1 for item in preview if not item.get("role_access", {}).get(role))

    log_sig = (role, query, len(retrieval["results"]), answer["answer_mode"])
    if st.session_state.get("last_rag_query_log") != log_sig:
        safe_log_rag_query(
            user_role=role, query=query, top_k=5,
            retrieved_count=len(retrieval["results"]), latency_ms=latency_ms,
            confidence_score=confidence_score,
            fallback_used=answer["answer_mode"] == "extractive",
            access_denied_count=access_denied_count,
            answer_mode=str(answer["answer_mode"]),
        )
        st.session_state["last_rag_query_log"] = log_sig

    # Layout
    ans_col, stat_col = st.columns([3, 1])

    with ans_col:
        section_header("◈", "Answer")

        # Query echo
        st.markdown(
            f'<div class="chat-query">{query}</div>',
            unsafe_allow_html=True,
        )

        # Answer
        mode_html = _answer_mode_html(answer["answer_mode"])
        st.markdown(
            f"""<div class="chat-answer">
                <div style="margin-bottom:10px">{mode_html}</div>
                <div style="color:#c9d1d9;line-height:1.7">{answer["answer"]}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Access rule
        st.markdown(
            f"""<div class="info-card" style="margin-top:12px">
                <div class="info-card-label">Access Rule Applied</div>
                <div style="font-size:0.8125rem;color:#8b949e">{retrieval["access_rule"]}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with stat_col:
        _confidence_gauge(confidence_score)

        # Stats
        st.markdown(
            f"""<div class="info-card">
                <div class="info-card-label">Query Statistics</div>
                <table style="width:100%;font-size:0.8125rem;border-collapse:collapse">
                    <tr><td style="color:#6e7681;padding:4px 0">Latency</td>
                        <td style="text-align:right;color:#e6edf3;font-weight:600">{latency_ms:.0f} ms</td></tr>
                    <tr><td style="color:#6e7681;padding:4px 0">Sources</td>
                        <td style="text-align:right;color:#e6edf3;font-weight:600">{len(retrieval["results"])}</td></tr>
                    <tr><td style="color:#6e7681;padding:4px 0">Blocked</td>
                        <td style="text-align:right;color:#f85149;font-weight:600">{access_denied_count}</td></tr>
                    <tr><td style="color:#6e7681;padding:4px 0">Role</td>
                        <td style="text-align:right;color:#58a6ff;font-weight:600">{role}</td></tr>
                </table>
            </div>""",
            unsafe_allow_html=True,
        )

    section_header("◻", "Source Citations")
    sources = _sources_table(retrieval["results"])
    if sources.empty:
        st.warning("No accessible sources were returned for this role and query.")
    else:
        _render_sources(retrieval["results"])
