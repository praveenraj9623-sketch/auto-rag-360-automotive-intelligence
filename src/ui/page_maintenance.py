"""
AutoRAG 360 — Predictive Maintenance
Professional gauge, 3D risk surface, feature importance.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ml.predict import predict_maintenance_risk
from src.ui.styles import PLOTLY_THEME, apply_plotly_theme, page_header, section_header


def _risk_result_block(risk_level: str, probability: float, recommendation: str) -> None:
    colors = {"High": "#f85149", "Medium": "#d29922", "Low": "#3fb950"}
    bgs = {"High": "rgba(248,81,73,0.08)", "Medium": "rgba(210,153,34,0.08)", "Low": "rgba(63,185,80,0.08)"}
    borders = {"High": "rgba(248,81,73,0.25)", "Medium": "rgba(210,153,34,0.25)", "Low": "rgba(63,185,80,0.25)"}
    color = colors.get(risk_level, "#6e7681")
    bg = bgs.get(risk_level, "rgba(110,118,129,0.08)")
    border = borders.get(risk_level, "rgba(110,118,129,0.25)")

    st.markdown(
        f"""<div style="background:{bg};border:1px solid {border};border-radius:8px;padding:20px;margin:12px 0">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
                <div>
                    <div style="font-size:0.6875rem;font-weight:600;color:#6e7681;
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">Risk Level</div>
                    <div style="font-size:1.75rem;font-weight:700;color:{color}">{risk_level}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.6875rem;font-weight:600;color:#6e7681;
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">Failure Probability</div>
                    <div style="font-size:2rem;font-weight:700;color:{color}">{probability:.1%}</div>
                </div>
            </div>
            <div style="background:rgba(0,0,0,0.3);border-radius:3px;height:4px;margin-bottom:12px">
                <div style="width:{min(probability*100, 100):.0f}%;height:100%;
                    background:{color};border-radius:3px"></div>
            </div>
            <div style="font-size:0.8125rem;color:#8b949e;line-height:1.6">{recommendation}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _failure_gauge(probability: float) -> None:
    t = PLOTLY_THEME
    color = t["danger"] if probability > 0.5 else t["warning"] if probability > 0.25 else t["success"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": color, "size": 32, "family": t["font_family"]},
                    "valueformat": ".1f"},
            title={"text": "Failure Probability", "font": {"size": 13, "color": "#8b949e", "family": t["font_family"]}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 9, "color": "#6e7681"}, "nticks": 6},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": t["paper_bg"],
                "bordercolor": "#30363d",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, 25], "color": "rgba(63,185,80,0.08)"},
                    {"range": [25, 50], "color": "rgba(210,153,34,0.06)"},
                    {"range": [50, 100], "color": "rgba(248,81,73,0.08)"},
                ],
                "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.8, "value": probability * 100},
            },
        )
    )
    fig = apply_plotly_theme(fig, "", height=260)
    fig.update_layout(margin=dict(l=16, r=16, t=30, b=16))
    st.plotly_chart(fig, use_container_width=True)


def _feature_importance(rows: list[dict]) -> None:
    if not rows:
        st.info("Feature importance unavailable — run the explainability step first.")
        return
    t = PLOTLY_THEME
    features = [r["feature"] for r in rows]
    importances = [float(r["importance"]) for r in rows]

    fig = go.Figure(
        go.Bar(
            x=importances,
            y=features,
            orientation="h",
            marker=dict(color=t["accent"], line=dict(width=0), opacity=0.8),
            text=[f"{v:.4f}" for v in importances],
            textposition="outside",
            textfont=dict(color="#8b949e", size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig = apply_plotly_theme(fig, "Feature Importance", height=max(200, len(rows) * 36))
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending", gridcolor="transparent"),
        xaxis=dict(gridcolor=t["grid_color"]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _risk_surface_3d(torque_val: float, tool_wear_val: float, air_temp_val: float) -> None:
    """3D surface: failure probability over torque × tool_wear space."""
    torque_range = np.linspace(5, 80, 30)
    wear_range = np.linspace(0, 250, 30)
    T, W = np.meshgrid(torque_range, wear_range)
    Z = 1 / (1 + np.exp(-(0.04 * T + 0.008 * W - 3.2 + 0.02 * (air_temp_val - 300))))
    Z = np.clip(Z, 0, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=torque_range,
            y=wear_range,
            z=Z,
            colorscale=[
                [0.0, "#0d1117"],
                [0.3, "#1d4ed8"],
                [0.6, "#d29922"],
                [1.0, "#f85149"],
            ],
            opacity=0.88,
            showscale=True,
            colorbar=dict(
                title=dict(text="Failure Prob", font=dict(color="#8b949e", size=11)),
                tickfont=dict(color="#8b949e", size=9),
                bgcolor="rgba(22,27,34,0.9)",
                bordercolor="#30363d",
                thickness=10,
            ),
            hovertemplate="Torque: %{x:.0f} Nm<br>Tool Wear: %{y:.0f} min<br>Prob: %{z:.2%}<extra></extra>",
        )
    )

    # Current input point
    z_cur = float(1 / (1 + np.exp(-(0.04 * torque_val + 0.008 * tool_wear_val - 3.2 + 0.02 * (air_temp_val - 300)))))
    fig.add_trace(
        go.Scatter3d(
            x=[torque_val], y=[tool_wear_val], z=[min(z_cur + 0.06, 1.0)],
            mode="markers+text",
            marker=dict(size=8, color="#58a6ff", symbol="diamond", line=dict(color="#ffffff", width=1)),
            text=["Your Input"],
            textfont=dict(color="#58a6ff", size=11),
            name="Current Input",
        )
    )

    def _axis(title):
        return dict(
            title=dict(text=title, font=dict(color="#8b949e", size=10)),
            backgroundcolor="#0d1117",
            gridcolor="#21262d",
            showbackground=True,
            tickfont=dict(color="#6e7681", size=9),
        )

    fig.update_layout(
        scene=dict(
            xaxis=_axis("Torque (Nm)"),
            yaxis=_axis("Tool Wear (min)"),
            zaxis=dict(**_axis("Failure Prob"), range=[0, 1]),
            bgcolor="#0d1117",
            camera=dict(eye=dict(x=1.7, y=-1.4, z=1.1)),
        ),
        paper_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="Inter"),
        height=460,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="3D Risk Surface — Torque × Tool Wear → Failure Probability",
                   font=dict(color="#8b949e", size=13), x=0.0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _recommendation(risk_level: str, probability: float) -> str:
    if risk_level == "High":
        return "Schedule immediate inspection. Prioritize thermal, torque, and tool-wear checks."
    if risk_level == "Medium":
        return "Monitor closely. Plan preventive maintenance in the next service window."
    return "Continue normal operations. Maintain routine inspection cadence."


def render_maintenance_page() -> None:
    page_header(
        "Predictive Maintenance · AI4I-Style Model",
        "Predictive Maintenance",
        "Machine failure classification using AI4I-style features. "
        "Adjust parameters to predict failure probability and explore the 3D risk surface.",
    )

    section_header("⚙️", "Machine Parameters")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        air_temperature = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process_temperature = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)
    with col_b:
        rotational_speed = st.slider("Rotational Speed (RPM)", 1100, 2900, 1500, 10)
        torque = st.slider("Torque (Nm)", 5.0, 80.0, 40.0, 0.5)
    with col_c:
        tool_wear = st.slider("Tool Wear (min)", 0, 250, 100, 1)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    input_payload = {
        "Air temperature": air_temperature,
        "Process temperature": process_temperature,
        "Rotational speed": rotational_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
        "Type": "M",
    }

    if predict_btn:
        st.session_state["total_predictions_made"] = st.session_state.get("total_predictions_made", 0) + 1
        with st.spinner("Running prediction model..."):
            st.session_state["maintenance_prediction"] = predict_maintenance_risk(input_payload)

    prediction = st.session_state.get("maintenance_prediction")

    if prediction:
        probability = float(prediction["failure_probability"])
        risk_level = prediction["risk_level"]

        res_col, gauge_col = st.columns([2, 1])
        with res_col:
            section_header("◆", "Prediction Result")
            _risk_result_block(risk_level, probability, _recommendation(risk_level, probability))
            section_header("▦", "Feature Importance")
            _feature_importance(prediction["top_contributing_features"])
        with gauge_col:
            section_header("◈", "Failure Gauge")
            _failure_gauge(probability)

    section_header("◻", "3D Risk Surface")
    st.caption(
        "Failure probability across the Torque × Tool Wear parameter space. "
        "Your current input is marked with a blue diamond."
    )
    _risk_surface_3d(torque, tool_wear, air_temperature)
