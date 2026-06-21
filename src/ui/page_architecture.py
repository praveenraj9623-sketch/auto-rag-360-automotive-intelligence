"""
AutoRAG 360 — Architecture Page
Clean visual system diagram, phase timeline, capability matrix.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from src.ui.styles import page_header, section_header


def _architecture_diagram() -> None:
    html = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        body { margin: 0; padding: 0; background: transparent; }
        .arch {
            font-family: 'Inter', -apple-system, sans-serif;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 24px;
            color: #c9d1d9;
        }
        .arch-title {
            font-size: 12px; font-weight: 600;
            color: #6e7681; text-transform: uppercase;
            letter-spacing: 0.08em; margin-bottom: 20px;
        }
        .arch-section { margin-bottom: 4px; }
        .arch-section-label {
            font-size: 11px; font-weight: 600;
            color: #6e7681; text-transform: uppercase;
            letter-spacing: 0.06em; margin: 16px 0 8px;
        }
        .arch-row {
            display: flex; align-items: stretch;
            justify-content: flex-start; gap: 0;
        }
        .arch-node {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 12px;
            font-weight: 500;
            color: #c9d1d9;
            min-width: 120px;
            max-width: 160px;
            flex: 1;
            transition: border-color 0.15s;
            cursor: default;
        }
        .arch-node:hover { border-color: #1f6feb; color: #e6edf3; }
        .arch-node-sub {
            font-size: 10px; color: #6e7681;
            font-weight: 400; display: block; margin-top: 3px;
        }
        .arch-node.accent { border-color: #1f6feb; color: #58a6ff; }
        .arch-node.success { border-color: #3fb950; color: #3fb950; }
        .arch-node.warn { border-color: #d29922; color: #d29922; }
        .arch-node.danger { border-color: #f85149; color: #f85149; }
        .arch-arrow {
            display: flex; align-items: center;
            padding: 0 10px; color: #30363d; font-size: 14px;
            flex-shrink: 0;
        }
        .arch-down {
            text-align: center; color: #30363d;
            font-size: 16px; padding: 5px 0;
        }
        .arch-full {
            background: #0d1117; border: 1px solid #30363d;
            border-radius: 6px; padding: 10px 16px;
            font-size: 12px; font-weight: 500; color: #c9d1d9;
        }
        .arch-full.accent { border-color: #1f6feb; color: #58a6ff; }
        .arch-full.danger { border-color: #f85149; color: #f85149; }
    </style>
    <div class="arch">
        <div class="arch-title">System Architecture — AutoRAG 360</div>

        <div class="arch-section-label">Data Ingestion</div>
        <div class="arch-row">
            <div class="arch-node">Recall CSVs<span class="arch-node-sub">NHTSA-style</span></div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">Complaint CSVs<span class="arch-node-sub">NHTSA-style</span></div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">Automotive Docs<span class="arch-node-sub">Service / Quality / Supplier</span></div>
        </div>
        <div class="arch-down">↓</div>

        <div class="arch-section-label">RAG Pipeline</div>
        <div class="arch-row">
            <div class="arch-node accent">Data Cleaning<span class="arch-node-sub">Pandas ETL</span></div>
            <div class="arch-arrow">→</div>
            <div class="arch-node accent">Chunking + TF-IDF<span class="arch-node-sub">Metadata tagging</span></div>
            <div class="arch-arrow">→</div>
            <div class="arch-node accent">RBAC Filter<span class="arch-node-sub">4 roles</span></div>
            <div class="arch-arrow">→</div>
            <div class="arch-node accent">Cosine Retrieval<span class="arch-node-sub">Top-K chunks</span></div>
        </div>
        <div class="arch-down">↓</div>

        <div class="arch-full accent">
            LLM Generation: Groq → Gemini → Extractive Fallback &nbsp; | &nbsp; Grounded answers with source citations
        </div>
        <div class="arch-down">↓</div>

        <div class="arch-section-label">Application Pages</div>
        <div class="arch-row">
            <div class="arch-node success">Executive Overview</div>
            <div class="arch-arrow">·</div>
            <div class="arch-node success">RAG Assistant</div>
            <div class="arch-arrow">·</div>
            <div class="arch-node success">Access Control</div>
            <div class="arch-arrow">·</div>
            <div class="arch-node success">Evaluation</div>
        </div>

        <div class="arch-section-label" style="margin-top:20px">ML Pipeline</div>
        <div class="arch-row">
            <div class="arch-node warn">AI4I Dataset<span class="arch-node-sub">Maintenance failure</span></div>
            <div class="arch-arrow">→</div>
            <div class="arch-node warn">UCI Automobile<span class="arch-node-sub">Vehicle risk</span></div>
            <div class="arch-arrow">→</div>
            <div class="arch-node warn">Complaint Text<span class="arch-node-sub">NLP severity</span></div>
        </div>
        <div class="arch-down">↓</div>

        <div class="arch-full danger">
            SQLite Monitoring DB — Query logs, prediction logs, latency, fallback rate, access denials
            &nbsp; | &nbsp; artifacts/monitoring.db
        </div>
    </div>
    """
    components.html(html, height=540, scrolling=False)


def _phase_cards() -> None:
    phases = [
        {
            "n": "1", "title": "Foundation", "color": "#1f6feb",
            "items": [
                "Streamlit RAG core + data ingestion",
                "Local TF-IDF retrieval + cosine scoring",
                "Role-based access control (4 roles)",
                "Extractive answer fallback",
                "Executive Overview + RAG Assistant",
            ],
        },
        {
            "n": "2", "title": "ML Layer", "color": "#d29922",
            "items": [
                "Predictive maintenance (AI4I-style)",
                "Complaint severity NLP classifier",
                "Vehicle risk scoring (UCI Automobile)",
                "Model artifacts, metrics, explainability",
                "Feature importance analysis",
            ],
        },
        {
            "n": "3", "title": "Production", "color": "#3fb950",
            "items": [
                "Optional Groq / Gemini LLM generation",
                "Golden-question RAG evaluation suite",
                "SQLite audit monitoring (9+ metrics)",
                "Architecture documentation",
                "Production-grade UI design system",
            ],
        },
    ]

    cols = st.columns(3)
    for col, phase in zip(cols, phases):
        with col:
            items_html = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px">'
                f'<span style="color:{phase["color"]};flex-shrink:0;font-size:0.75rem;margin-top:2px">✓</span>'
                f'<span style="font-size:0.8125rem;color:#8b949e;line-height:1.4">{item}</span>'
                f'</div>'
                for item in phase["items"]
            )
            st.markdown(
                f"""<div style="background:#161b22;border:1px solid #21262d;
                    border-top:2px solid {phase['color']};border-radius:8px;padding:16px 18px">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
                        <div style="width:24px;height:24px;border-radius:4px;
                            background:{phase['color']};display:flex;align-items:center;
                            justify-content:center;font-weight:700;font-size:0.8rem;
                            color:#0d1117;flex-shrink:0">{phase['n']}</div>
                        <div>
                            <div style="font-size:0.6875rem;color:#6e7681;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.06em">Phase {phase['n']}</div>
                            <div style="font-size:0.9375rem;font-weight:600;color:{phase['color']}">{phase['title']}</div>
                        </div>
                    </div>
                    {items_html}
                </div>""",
                unsafe_allow_html=True,
            )


def _jd_table() -> None:
    rows = [
        ("RAG / Semantic Search", "TF-IDF retrieval over recalls, complaints, and automotive docs"),
        ("Responsible AI / Grounded Outputs", "Cited answer modes with extractive fallback"),
        ("Access Control", "4-role document_type and sensitivity_level filtering"),
        ("ML Modeling", "Maintenance, severity, and vehicle risk classification"),
        ("Evaluation", "Golden-question Precision@K, Hit Rate, MRR, latency"),
        ("Monitoring", "SQLite query and prediction audit logs"),
        ("Executive Reporting", "KPI cards, model health gauges, prediction counters"),
        ("Cloud Deployment", "Streamlit Cloud free-tier, lightweight dependencies"),
    ]
    for capability, implementation in rows:
        st.markdown(
            f"""<div class="source-item" style="margin-bottom:4px">
                <span style="font-size:0.875rem;font-weight:600;color:#c9d1d9;flex:0 0 280px">{capability}</span>
                <span style="font-size:0.8125rem;color:#8b949e">{implementation}</span>
                <span style="color:#3fb950;flex-shrink:0">✓</span>
            </div>""",
            unsafe_allow_html=True,
        )


def _dataset_table() -> None:
    datasets = [
        ("NHTSA-style Recalls & Complaints", "RAG corpus and complaint analytics", "Fallback unless replaced with real NHTSA exports"),
        ("AI4I 2020 Predictive Maintenance", "Machine failure classification", "Real file used when present; synthetic fallback if absent"),
        ("UCI Automobile", "Vehicle risk category model", "UCI-style raw data normalized into automobile.csv"),
        ("Sample Automotive Documents", "Service, quality, recall policy, supplier retrieval", "Project-created Phase 1 sample documents"),
    ]
    for name, use, note in datasets:
        st.markdown(
            f"""<div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
                padding:12px 16px;margin-bottom:6px">
                <div style="font-size:0.875rem;font-weight:600;color:#c9d1d9;margin-bottom:3px">{name}</div>
                <div style="font-size:0.8125rem;color:#6e7681">
                    <span style="color:#58a6ff">Use:</span> {use} &nbsp;·&nbsp;
                    <span style="color:#6e7681">{note}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_architecture_page() -> None:
    page_header(
        "Architecture · 3-Phase Design",
        "System Architecture",
        "AutoRAG 360 delivers a governed automotive intelligence platform using RAG, "
        "role-based access control, ML risk models, evaluation, and SQLite audit monitoring — "
        "all deployable on Streamlit Cloud free tier.",
    )

    section_header("◻", "System Architecture Diagram")
    _architecture_diagram()

    section_header("▦", "Built in 3 Phases")
    _phase_cards()

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("◆", "JD Capability Mapping")
    _jd_table()

    section_header("◻", "Dataset Sources")
    _dataset_table()

    st.divider()
    section_header("⚙️", "Technology Stack")
    stack = [
        ("Streamlit", "≥1.35", "Web application framework"),
        ("Pandas", "≥2.0", "Data loading and analytics"),
        ("NumPy", "≥1.24", "Numerical computation"),
        ("scikit-learn", "≥1.3", "TF-IDF, classifiers, KNN"),
        ("Plotly", "≥5.18", "Interactive and 3D charts"),
        ("SQLite3", "stdlib", "Zero-infrastructure audit logs"),
        ("pytest", "≥8.0", "Test suite"),
    ]
    cols = st.columns(4)
    for i, (name, version, desc) in enumerate(stack):
        with cols[i % 4]:
            st.markdown(
                f"""<div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
                    padding:12px 14px;margin-bottom:10px">
                    <div style="font-weight:600;color:#e6edf3;font-size:0.875rem">{name}</div>
                    <div style="font-size:0.75rem;color:#58a6ff;font-family:monospace;margin:2px 0">{version}</div>
                    <div style="font-size:0.75rem;color:#6e7681;line-height:1.4">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )
