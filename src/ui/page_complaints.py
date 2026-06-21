"""
AutoRAG 360 — Complaint Severity
Professional NLP classifier UI with sunburst, treemap, bar charts.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from src.data.load_datasets import load_complaints
from src.ml.predict import predict_complaint_severity
from src.ui.styles import PLOTLY_THEME, apply_plotly_theme, page_header, section_header


def _top_issue_keywords(complaints: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    texts = complaints["CDESCR"].fillna("").astype(str)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=limit)
    matrix = vectorizer.fit_transform(texts)
    counts = matrix.sum(axis=0).A1
    rows = sorted(zip(vectorizer.get_feature_names_out(), counts), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(rows, columns=["keyword", "count"])


def _severity_bars(probabilities: dict) -> None:
    t = PLOTLY_THEME
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors_map = {"High": t["danger"], "Medium": t["warning"], "Low": t["success"]}
    colors = [colors_map.get(l, t["accent"]) for l in labels]

    fig = go.Figure(
        go.Bar(
            x=labels, y=values,
            marker=dict(color=colors, line=dict(width=0), opacity=0.85),
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            textfont=dict(color=t["font_color"], size=12),
            hovertemplate="<b>%{x}</b><br>%{y:.2%}<extra></extra>",
        )
    )
    fig = apply_plotly_theme(fig, "Severity Probability", height=260)
    fig.update_layout(
        xaxis=dict(gridcolor="transparent"),
        yaxis=dict(range=[0, max(values, default=0.1) * 1.35], gridcolor=t["grid_color"],
                   tickformat=".0%"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _complaint_sunburst(complaints: pd.DataFrame) -> None:
    t = PLOTLY_THEME
    counts = complaints["COMPONENT"].value_counts().head(10)

    labels = ["All Complaints"] + [str(c) for c in counts.index]
    parents = [""] + ["All Complaints"] * len(counts)
    values = [int(complaints.shape[0])] + [int(v) for v in counts.values]

    fig = go.Figure(
        go.Sunburst(
            labels=labels, parents=parents, values=values,
            branchvalues="total",
            marker=dict(colors=["#161b22"] + t["categorical"][:len(counts)],
                        line=dict(color=t["bg"], width=1.5)),
            textfont=dict(family=t["font_family"], size=11),
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>%{value} complaints<br>%{percentRoot:.1%}<extra></extra>",
        )
    )
    fig = apply_plotly_theme(fig, "Complaint Hierarchy by Component", height=380)
    fig.update_layout(margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _keyword_treemap(keywords_df: pd.DataFrame) -> None:
    t = PLOTLY_THEME
    fig = px.treemap(
        keywords_df, path=["keyword"], values="count",
        color="count",
        color_continuous_scale=[[0, "#21262d"], [1.0, t["accent"]]],
    )
    fig.update_traces(
        textinfo="label+value",
        textfont=dict(size=11, family=t["font_family"]),
        hovertemplate="<b>%{label}</b><br>Frequency: %{value}<extra></extra>",
    )
    fig = apply_plotly_theme(fig, "Top Issue Keywords", height=320)
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=4, r=4, t=48, b=4))
    st.plotly_chart(fig, use_container_width=True)


def _component_bar(complaints: pd.DataFrame) -> None:
    t = PLOTLY_THEME
    top = complaints["COMPONENT"].value_counts().head(12).reset_index()
    top.columns = ["component", "count"]

    fig = go.Figure(
        go.Bar(
            x=top["count"], y=top["component"], orientation="h",
            marker=dict(color=t["danger"], line=dict(width=0), opacity=0.75),
            text=top["count"], textposition="outside",
            textfont=dict(color=t["font_color"], size=10),
            hovertemplate="<b>%{y}</b><br>Complaints: %{x}<extra></extra>",
        )
    )
    fig = apply_plotly_theme(fig, "Complaints by Component", height=380)
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending", gridcolor="transparent"),
        xaxis=dict(gridcolor=t["grid_color"]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_complaints_page() -> None:
    page_header(
        "Complaint Severity · TF-IDF Classifier",
        "Complaint Severity",
        "Rule-labeled TF-IDF logistic regression classifier. "
        "Submit complaint text to classify severity and inspect contributing terms.",
    )

    section_header("▦", "Severity Classifier")
    inp_col, res_col = st.columns([3, 2])

    with inp_col:
        complaint_text = st.text_area(
            "Complaint Text",
            value="Vehicle experienced brake failure warning and reduced stopping ability while driving.",
            height=130,
            placeholder="Paste or type a complaint description...",
        )
        predict_btn = st.button("Classify Severity", type="primary")

    with res_col:
        if predict_btn and complaint_text.strip():
            st.session_state["total_predictions_made"] = st.session_state.get("total_predictions_made", 0) + 1
            with st.spinner("Classifying..."):
                st.session_state["complaint_prediction"] = predict_complaint_severity(complaint_text)

        pred = st.session_state.get("complaint_prediction")
        if pred:
            sev = pred["severity"]
            conf = pred["confidence"]
            sev_colors = {"High": "#f85149", "Medium": "#d29922", "Low": "#3fb950"}
            sev_bgs = {"High": "rgba(248,81,73,0.08)", "Medium": "rgba(210,153,34,0.08)", "Low": "rgba(63,185,80,0.08)"}
            sev_borders = {"High": "rgba(248,81,73,0.25)", "Medium": "rgba(210,153,34,0.25)", "Low": "rgba(63,185,80,0.25)"}
            color = sev_colors.get(sev, "#8b949e")
            bg = sev_bgs.get(sev, "rgba(110,118,129,0.08)")
            border = sev_borders.get(sev, "rgba(110,118,129,0.25)")

            st.markdown(
                f"""<div style="background:{bg};border:1px solid {border};border-radius:8px;
                    padding:20px;text-align:center">
                    <div style="font-size:0.6875rem;font-weight:600;color:#6e7681;
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">Predicted Severity</div>
                    <div style="font-size:2.25rem;font-weight:700;color:{color};margin-bottom:6px">{sev}</div>
                    <div style="font-size:0.8125rem;color:#8b949e">{conf:.1%} confidence</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Probability & terms
    pred = st.session_state.get("complaint_prediction")
    if pred:
        prob_col, terms_col = st.columns([3, 2])
        with prob_col:
            _severity_bars(pred["class_probabilities"])
        with terms_col:
            if pred["top_contributing_features"]:
                st.markdown(
                    """<div class="info-card">
                        <div class="info-card-label">Top Contributing Terms</div>""",
                    unsafe_allow_html=True,
                )
                for feat in pred["top_contributing_features"][:8]:
                    imp = float(feat.get("importance", 0))
                    bar_w = min(100, int(imp * 300))
                    st.markdown(
                        f"""<div style="margin:7px 0">
                            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                                <span style="font-size:0.8125rem;color:#8b949e;font-family:monospace">
                                    "{feat.get('feature','')}"</span>
                                <span style="font-size:0.8125rem;font-weight:600;color:#58a6ff;
                                    font-family:monospace">{imp:.4f}</span>
                            </div>
                            <div style="background:#21262d;border-radius:2px;height:3px">
                                <div style="width:{bar_w}%;height:100%;background:#1f6feb;border-radius:2px"></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

    # Dataset analytics
    st.divider()
    section_header("📊", "Complaint Dataset Analytics")
    complaints = load_complaints()

    col1, col2 = st.columns([1, 1])
    with col1:
        _complaint_sunburst(complaints)
    with col2:
        _component_bar(complaints)

    section_header("▦", "Issue Keyword Map")
    _keyword_treemap(_top_issue_keywords(complaints, 20))
