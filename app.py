from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.data.load_datasets import load_dashboard_data
from src.monitoring.audit_logger import initialize_database
from src.rag.retriever import build_default_retriever
from src.ui.page_access_control import render_access_control_demo
from src.ui.page_architecture import render_architecture_page
from src.ui.page_complaints import render_complaints_page
from src.ui.page_evaluation import render_evaluation_page
from src.ui.page_maintenance import render_maintenance_page
from src.ui.page_monitoring import render_monitoring_page
from src.ui.page_overview import render_overview
from src.ui.page_rag_assistant import render_rag_assistant
from src.ui.page_vehicle_risk import render_vehicle_risk_page
from src.ui.styles import inject_global_css


st.set_page_config(
    page_title="AutoRAG 360 | Automotive Intelligence Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AutoRAG 360 — Automotive Intelligence Platform. RAG + ML + RBAC.",
    },
)

inject_global_css()


@st.cache_data(show_spinner=False)
def get_dashboard_data():
    return load_dashboard_data()


@st.cache_resource(show_spinner="Loading corpus index...")
def get_retriever():
    return build_default_retriever()


def get_maintenance_model_health() -> dict | None:
    metrics_path = (
        Path(__file__).resolve().parent / "artifacts" / "reports" / "maintenance_metrics.json"
    )
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


PAGES = {
    "Executive Overview": "Executive Overview",
    "RAG Assistant": "Automotive RAG Assistant",
    "Access Control Demo": "Access Control Demo",
    "Predictive Maintenance": "Predictive Maintenance",
    "Vehicle Risk Scoring": "Vehicle Risk Scoring",
    "Complaint Severity": "Complaint Severity",
    "RAG Evaluation": "Evaluation",
    "Monitoring": "Monitoring",
    "Architecture": "Architecture",
}

PAGE_ICONS = {
    "Executive Overview": "⬜",
    "RAG Assistant": "◈",
    "Access Control Demo": "◉",
    "Predictive Maintenance": "◆",
    "Vehicle Risk Scoring": "◇",
    "Complaint Severity": "△",
    "RAG Evaluation": "▦",
    "Monitoring": "▣",
    "Architecture": "▤",
}


def render_sidebar() -> str:
    with st.sidebar:
        # Logo block
        st.markdown(
            """<div class="sb-logo">
                <div class="sb-logo-name">AutoRAG 360</div>
                <div class="sb-logo-sub">Automotive Intelligence Platform</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section-label">Navigation</div>', unsafe_allow_html=True)

        selected = st.radio(
            "nav",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Status
        st.markdown(
            """<div class="sb-status-row">
                <div class="sb-status-dot"></div>
                <span style="color:#3fb950 !important;font-weight:600">System Online</span>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """<div style="padding:4px 12px 12px; font-size:0.75rem; color:#6e7681 !important;">
                <div style="margin:3px 0; color:#6e7681 !important;">TF-IDF RAG · RBAC · SQLite Audit</div>
                <div style="margin:3px 0; color:#6e7681 !important;">Groq → Gemini → Extractive fallback</div>
                <div style="margin:3px 0; color:#6e7681 !important;">Streamlit Cloud compatible</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            """<div style="padding:0 12px 8px; font-size:0.6875rem; color:#6e7681 !important;">
                v3.0 · Phase 1–3 Complete
            </div>""",
            unsafe_allow_html=True,
        )

    return PAGES[selected]


def main() -> None:
    initialize_database()
    st.session_state.setdefault("total_predictions_made", 0)

    page = render_sidebar()

    recalls, complaints, data_source_label = get_dashboard_data()
    retriever = get_retriever()

    if page == "Executive Overview":
        render_overview(
            recalls,
            complaints,
            data_source_label,
            maintenance_health=get_maintenance_model_health(),
            prediction_count=st.session_state["total_predictions_made"],
        )
    elif page == "Automotive RAG Assistant":
        render_rag_assistant(retriever, data_source_label)
    elif page == "Access Control Demo":
        render_access_control_demo(retriever, data_source_label)
    elif page == "Predictive Maintenance":
        render_maintenance_page()
    elif page == "Vehicle Risk Scoring":
        render_vehicle_risk_page()
    elif page == "Complaint Severity":
        render_complaints_page()
    elif page == "Evaluation":
        render_evaluation_page()
    elif page == "Monitoring":
        render_monitoring_page()
    elif page == "Architecture":
        render_architecture_page()


if __name__ == "__main__":
    main()
