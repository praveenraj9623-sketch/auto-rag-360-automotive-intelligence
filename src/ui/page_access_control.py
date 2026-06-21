"""
AutoRAG 360 — Access Control Demo
Professional side-by-side role comparison.
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.monitoring.audit_logger import safe_log_rag_query
from src.rag.access_control import describe_policy, get_roles
from src.rag.generator import generate_grounded_answer
from src.ui.page_overview import render_data_source_badge
from src.ui.styles import PLOTLY_THEME, apply_plotly_theme, page_header, section_header


def _access_preview_rows(preview: list[dict], role: str) -> pd.DataFrame:
    rows = []
    for item in preview:
        meta = item.get("metadata", {})
        rows.append({
            "visibility": "Accessible" if item.get("role_access", {}).get(role) else "Blocked",
            "source": meta.get("source", "unknown"),
            "document_type": meta.get("document_type", "unknown"),
            "sensitivity_level": meta.get("sensitivity_level", "unknown"),
        })
    return pd.DataFrame(rows)


def _visibility_donut(can_see: int, blocked: int) -> None:
    t = PLOTLY_THEME
    total = can_see + blocked
    if total == 0:
        can_see, total = 0, 1
    fig = go.Figure(
        go.Pie(
            labels=["Accessible", "Blocked"],
            values=[can_see, blocked] if (can_see + blocked) > 0 else [1, 0],
            hole=0.62,
            marker=dict(colors=[t["success"], t["danger"]], line=dict(color=t["bg"], width=2)),
            textinfo="percent",
            textfont=dict(size=11, family=t["font_family"]),
            hovertemplate="<b>%{label}</b><br>%{value}<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"{can_see}/{can_see+blocked}<br><span style='font-size:8px'>visible</span>",
        x=0.5, y=0.5,
        font=dict(size=12, color=t["success"] if can_see > 0 else "#8b949e"),
        showarrow=False,
    )
    fig = apply_plotly_theme(fig, "", height=180)
    fig.update_layout(
        showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _mode_badge(mode: str) -> str:
    styles = {
        "extractive": ("yellow", "Extractive"),
        "groq": ("green", "Groq LLM"),
        "gemini": ("blue", "Gemini LLM"),
    }
    s, label = styles.get(mode, ("gray", mode.capitalize()))
    return f'<span class="badge badge-{s}">{label}</span>'


def _role_panel(role: str, answer: dict, results: list[dict],
                preview_rows: pd.DataFrame, latency_ms: float, side_color: str) -> None:
    can_see = int((preview_rows["visibility"] == "Accessible").sum()) if not preview_rows.empty else 0
    blocked = int((preview_rows["visibility"] == "Blocked").sum()) if not preview_rows.empty else 0

    # Role header
    st.markdown(
        f"""<div style="background:#161b22;border:1px solid #21262d;
            border-top:2px solid {side_color};border-radius:8px;padding:14px 16px;margin-bottom:12px">
            <div style="font-size:0.6875rem;color:#6e7681;font-weight:600;
                text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">Role</div>
            <div style="font-size:1.125rem;font-weight:700;color:#e6edf3;margin-bottom:6px">{role}</div>
            <div style="font-size:0.8125rem;color:#8b949e;line-height:1.5">{describe_policy(role)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Visibility donut
    _visibility_donut(can_see, blocked)

    # Quick stats
    st.markdown(
        f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px">
            <div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
                padding:10px;text-align:center">
                <div style="font-size:1.25rem;font-weight:700;color:{PLOTLY_THEME['success']}">{can_see}</div>
                <div style="font-size:0.6875rem;color:#6e7681;text-transform:uppercase;
                    letter-spacing:0.05em">Visible</div>
            </div>
            <div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
                padding:10px;text-align:center">
                <div style="font-size:1.25rem;font-weight:700;color:{PLOTLY_THEME['danger']}">{blocked}</div>
                <div style="font-size:0.6875rem;color:#6e7681;text-transform:uppercase;
                    letter-spacing:0.05em">Blocked</div>
            </div>
            <div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
                padding:10px;text-align:center">
                <div style="font-size:1.25rem;font-weight:700;color:#58a6ff">{latency_ms:.0f}</div>
                <div style="font-size:0.6875rem;color:#6e7681;text-transform:uppercase;
                    letter-spacing:0.05em">ms</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Answer
    st.markdown(
        f"""<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
            padding:14px 16px;margin-bottom:12px">
            <div style="margin-bottom:10px">{_mode_badge(answer['answer_mode'])}</div>
            <div style="font-size:0.875rem;color:#c9d1d9;line-height:1.7">{answer['answer']}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Top sources
    for result in results[:4]:
        meta = result.get("metadata", {})
        source = meta.get("source", "unknown")
        doc_type = meta.get("document_type", "unknown").replace("_", " ").title()
        sensitivity = meta.get("sensitivity_level", "unknown")
        score = float(result.get("score", 0.0))
        s_colors = {"high": "#f85149", "medium": "#d29922", "low": "#3fb950"}
        s_color = s_colors.get(sensitivity.lower(), "#6e7681")
        st.markdown(
            f"""<div class="source-item">
                <div style="flex:1;min-width:0">
                    <div style="font-weight:500;color:#c9d1d9;font-size:0.8125rem;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{source}</div>
                    <div style="color:#6e7681;font-size:0.75rem">
                        {doc_type} · <span style="color:{s_color}">{sensitivity.upper()}</span>
                    </div>
                </div>
                <span style="font-size:0.75rem;font-family:monospace;color:#6e7681">{score:.4f}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def render_access_control_demo(retriever, data_source_label: str) -> None:
    page_header(
        "Access Control Demo · RBAC",
        "Access Control Demo",
        "Compare how two different roles see the same query — different document visibility, "
        "filtered sources, and grounded answers.",
    )
    render_data_source_badge(data_source_label)

    roles = get_roles()
    query = st.text_input(
        "Shared Query",
        value="brake actuator service manual supplier notice",
        key="access_control_query",
    )

    col_a, col_b = st.columns(2)
    role_a = col_a.selectbox("Left Role", roles, index=2, key="role_a")
    role_b = col_b.selectbox("Right Role", roles, index=3, key="role_b")

    preview = retriever.preview_access(query=query, roles=[role_a, role_b], top_k=8)

    left_start = time.perf_counter()
    left_results = retriever.retrieve(query=query, role=role_a, top_k=5)
    left_latency = (time.perf_counter() - left_start) * 1000

    right_start = time.perf_counter()
    right_results = retriever.retrieve(query=query, role=role_b, top_k=5)
    right_latency = (time.perf_counter() - right_start) * 1000

    left_answer = generate_grounded_answer(query, left_results["results"])
    right_answer = generate_grounded_answer(query, right_results["results"])

    left_denied = sum(1 for item in preview if not item.get("role_access", {}).get(role_a))
    right_denied = sum(1 for item in preview if not item.get("role_access", {}).get(role_b))

    for log_key, role, retrieval, answer, latency_ms, denied in [
        ("last_left_log", role_a, left_results, left_answer, left_latency, left_denied),
        ("last_right_log", role_b, right_results, right_answer, right_latency, right_denied),
    ]:
        sig = (role, query, len(retrieval["results"]), answer["answer_mode"])
        if st.session_state.get(log_key) != sig:
            safe_log_rag_query(
                user_role=role, query=query, top_k=5,
                retrieved_count=len(retrieval["results"]), latency_ms=latency_ms,
                confidence_score=max((float(r.get("score", 0.0)) for r in retrieval["results"]), default=0.0),
                fallback_used=answer["answer_mode"] == "extractive",
                access_denied_count=denied, answer_mode=str(answer["answer_mode"]),
            )
            st.session_state[log_key] = sig

    section_header("◉", "Side-by-Side Comparison")
    left_col, right_col = st.columns(2)

    left_preview = _access_preview_rows(preview, role_a)
    right_preview = _access_preview_rows(preview, role_b)

    with left_col:
        _role_panel(role_a, left_answer, left_results["results"], left_preview, left_latency, "#1f6feb")
    with right_col:
        _role_panel(role_b, right_answer, right_results["results"], right_preview, right_latency, "#8b949e")
