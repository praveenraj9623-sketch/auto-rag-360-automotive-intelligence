"""
AutoRAG 360 — Vehicle Risk Scoring
3D scatter, radar chart, risk gauge, professional layout.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ml.predict import predict_vehicle_risk, similar_vehicle_profiles
from src.ml.train_vehicle_risk import load_automobile_data
from src.ui.styles import PLOTLY_THEME, apply_plotly_theme, page_header, section_header


def _risk_gauge(confidence: float, risk_category: str) -> None:
    t = PLOTLY_THEME
    risk_colors = {
        "Low": t["success"], "Medium-Low": "#56d364",
        "Medium": t["warning"], "Medium-High": "#fb923c", "High": t["danger"],
    }
    color = risk_colors.get(risk_category, "#8b949e")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"color": color, "size": 28, "family": t["font_family"]},
                    "valueformat": ".1f"},
            title={"text": f"Confidence · {risk_category}",
                   "font": {"size": 11, "color": "#8b949e", "family": t["font_family"]}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 9, "color": "#6e7681"}, "nticks": 5},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": t["paper_bg"],
                "bordercolor": "#30363d",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, 50], "color": "rgba(110,118,129,0.06)"},
                    {"range": [50, 80], "color": "rgba(210,153,34,0.06)"},
                    {"range": [80, 100], "color": "rgba(63,185,80,0.06)"},
                ],
                "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.8, "value": confidence * 100},
            },
        )
    )
    fig = apply_plotly_theme(fig, "", height=220)
    fig.update_layout(margin=dict(l=16, r=16, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _vehicle_radar(input_payload: dict, automobile: pd.DataFrame) -> None:
    features = {
        "Horsepower": ("horsepower", input_payload.get("horsepower", 110)),
        "Curb Weight": ("curb-weight", input_payload.get("curb-weight", 2500)),
        "City MPG": ("city-mpg", input_payload.get("city-mpg", 24)),
        "Highway MPG": ("highway-mpg", input_payload.get("highway-mpg", 30)),
    }
    maxes = {}
    means = {}
    for label, (col, val) in features.items():
        if col in automobile.columns:
            nc = pd.to_numeric(automobile[col], errors="coerce")
            maxes[label] = float(nc.max()) or 1
            means[label] = float(nc.mean())
        else:
            maxes[label] = val * 2 or 1
            means[label] = val

    cats = list(features.keys())
    def norm(vals):
        return [v / maxes[c] if maxes[c] > 0 else 0 for v, c in zip(vals, cats)]

    inp_norm = norm([features[c][1] for c in cats])
    avg_norm = norm([means[c] for c in cats])
    cats_c = cats + [cats[0]]

    t = PLOTLY_THEME
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=inp_norm + [inp_norm[0]], theta=cats_c, fill="toself",
        name="Your Vehicle",
        line=dict(color=t["accent"], width=2),
        fillcolor="rgba(31,111,235,0.12)",
        marker=dict(size=5, color=t["accent"]),
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_norm + [avg_norm[0]], theta=cats_c, fill="toself",
        name="Dataset Average",
        line=dict(color="#8b949e", width=1.5, dash="dot"),
        fillcolor="rgba(139,148,158,0.06)",
        marker=dict(size=4, color="#8b949e"),
    ))
    fig = apply_plotly_theme(fig, "Vehicle Profile vs. Dataset Average", height=320)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#6e7681", size=9),
                            gridcolor="#21262d", showline=False),
            angularaxis=dict(tickfont=dict(color="#8b949e", size=10), gridcolor="#21262d"),
            bgcolor="#161b22",
        ),
        legend=dict(font=dict(color="#8b949e", size=10), bgcolor="transparent"),
        margin=dict(l=20, r=20, t=48, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _vehicle_3d_scatter(automobile: pd.DataFrame, input_payload: dict, risk_category: str) -> None:
    required = ["horsepower", "curb-weight", "city-mpg"]
    if not all(c in automobile.columns for c in required):
        st.info("3D scatter requires horsepower, curb-weight, and city-mpg columns.")
        return

    df = automobile.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    if df.empty:
        return

    color_col = "normalized-losses" if "normalized-losses" in df.columns else "horsepower"
    df[color_col] = pd.to_numeric(df[color_col], errors="coerce")
    color_vals = df[color_col].fillna(df[color_col].median())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["horsepower"], y=df["curb-weight"], z=df["city-mpg"],
            mode="markers",
            marker=dict(
                size=4,
                color=color_vals,
                colorscale=[[0, "#1d4ed8"], [0.5, "#d29922"], [1.0, "#f85149"]],
                colorbar=dict(
                    title=dict(text="Risk Score", font=dict(color="#8b949e", size=10)),
                    tickfont=dict(color="#8b949e", size=9),
                    bgcolor="rgba(22,27,34,0.9)",
                    bordercolor="#30363d",
                    thickness=8,
                ),
                opacity=0.65, line=dict(width=0),
            ),
            text=df.get("make", pd.Series([""] * len(df))),
            hovertemplate="<b>%{text}</b><br>HP: %{x}<br>Weight: %{y} lbs<br>City MPG: %{z}<extra></extra>",
            name="All Vehicles",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[input_payload.get("horsepower", 110)],
            y=[input_payload.get("curb-weight", 2500)],
            z=[input_payload.get("city-mpg", 24)],
            mode="markers+text",
            marker=dict(size=10, color="#58a6ff", symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            text=[f"Your Vehicle ({risk_category})"],
            textfont=dict(color="#58a6ff", size=10),
            name="Your Vehicle",
        )
    )

    def _ax(title):
        return dict(
            title=dict(text=title, font=dict(color="#8b949e", size=10)),
            backgroundcolor="#0d1117",
            gridcolor="#21262d",
            showbackground=True,
            tickfont=dict(color="#6e7681", size=9),
        )

    fig.update_layout(
        scene=dict(
            xaxis=_ax("Horsepower"),
            yaxis=_ax("Curb Weight (lbs)"),
            zaxis=_ax("City MPG"),
            bgcolor="#0d1117",
            camera=dict(eye=dict(x=1.7, y=-1.4, z=0.9)),
        ),
        paper_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="Inter"),
        height=460,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="3D Vehicle Space — Horsepower × Curb Weight × City MPG",
                   font=dict(color="#8b949e", size=13), x=0.0),
        legend=dict(font=dict(color="#8b949e", size=10), bgcolor="rgba(22,27,34,0.8)",
                    bordercolor="#30363d", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_vehicle_risk_page() -> None:
    page_header(
        "Vehicle Risk Scoring · UCI Automobile Model",
        "Vehicle Risk Scoring",
        "Normalized-loss risk category model. Configure a vehicle to score its risk category "
        "and explore the 3D vehicle feature space.",
    )

    automobile = load_automobile_data()
    makes = sorted(automobile["make"].unique())
    body_styles = sorted(automobile["body-style"].unique())
    engine_types = sorted(automobile["engine-type"].unique())
    fuel_systems = sorted(automobile["fuel-system"].unique())

    section_header("⚙️", "Vehicle Configuration")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        make = st.selectbox("Make", makes, index=makes.index("toyota") if "toyota" in makes else 0)
        body_style = st.selectbox("Body Style", body_styles,
                                  index=body_styles.index("sedan") if "sedan" in body_styles else 0)
        engine_type = st.selectbox("Engine Type", engine_types,
                                   index=engine_types.index("ohc") if "ohc" in engine_types else 0)
    with col_b:
        horsepower = st.number_input("Horsepower", min_value=40, max_value=300, value=110, step=5)
        curb_weight = st.number_input("Curb Weight (lbs)", min_value=1400, max_value=5000, value=2500, step=50)
        fuel_system = st.selectbox("Fuel System", fuel_systems,
                                   index=fuel_systems.index("mpfi") if "mpfi" in fuel_systems else 0)
    with col_c:
        city_mpg = st.number_input("City MPG", min_value=8, max_value=60, value=24, step=1)
        highway_mpg = st.number_input("Highway MPG", min_value=10, max_value=70, value=30, step=1)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Score Vehicle Risk", type="primary", use_container_width=True)

    input_payload = {
        "make": make, "body-style": body_style, "engine-type": engine_type,
        "horsepower": horsepower, "curb-weight": curb_weight,
        "fuel-system": fuel_system, "city-mpg": city_mpg, "highway-mpg": highway_mpg,
    }

    if predict_btn:
        st.session_state["total_predictions_made"] = st.session_state.get("total_predictions_made", 0) + 1
        with st.spinner("Computing risk score..."):
            st.session_state["vehicle_prediction"] = predict_vehicle_risk(input_payload)
            st.session_state["vehicle_neighbors"] = similar_vehicle_profiles(input_payload)

    prediction = st.session_state.get("vehicle_prediction")
    risk_category = prediction["risk_category"] if prediction else "Unknown"

    # 3D scatter always visible
    section_header("◻", "3D Vehicle Feature Space")
    _vehicle_3d_scatter(automobile, input_payload, risk_category)

    if prediction:
        res_col, radar_col = st.columns([1, 1])
        with res_col:
            section_header("◆", "Risk Score Result")
            _risk_gauge(prediction["confidence"], prediction["risk_category"])

            risk_colors = {
                "Low": "#3fb950", "Medium-Low": "#56d364",
                "Medium": "#d29922", "Medium-High": "#fb923c", "High": "#f85149",
            }
            color = risk_colors.get(prediction["risk_category"], "#8b949e")

            # Feature importance bars
            st.markdown(
                f"""<div class="info-card" style="margin-top:8px">
                    <div class="info-card-label">Contributing Features</div>""",
                unsafe_allow_html=True,
            )
            for feat in prediction.get("top_contributing_features", [])[:6]:
                imp = float(feat.get("importance", 0))
                bar_w = min(100, int(imp * 500))
                st.markdown(
                    f"""<div style="margin:8px 0">
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                            <span style="font-size:0.8125rem;color:#8b949e">{feat.get('feature','')}</span>
                            <span style="font-size:0.8125rem;font-weight:600;color:{color};
                                font-family:monospace">{imp:.4f}</span>
                        </div>
                        <div style="background:#21262d;border-radius:2px;height:3px">
                            <div style="width:{bar_w}%;height:100%;background:{color};border-radius:2px"></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with radar_col:
            section_header("◈", "Vehicle Profile Radar")
            _vehicle_radar(input_payload, automobile)

        section_header("◻", "Similar Vehicles")
        neighbors = st.session_state.get("vehicle_neighbors", pd.DataFrame())
        if not neighbors.empty:
            st.dataframe(neighbors, use_container_width=True, hide_index=True)
