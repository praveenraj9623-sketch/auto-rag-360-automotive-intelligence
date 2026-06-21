"""
AutoRAG 360 — Professional Design System
Enterprise-grade dark theme (GitHub-dark inspired).
Clean, readable, professional — suitable for senior technical audiences.
"""

from __future__ import annotations

import streamlit as st


# ─── Plotly theme tokens ──────────────────────────────────────────────────────

PLOTLY_THEME = {
    "bg": "#0d1117",
    "paper_bg": "#161b22",
    "grid_color": "#21262d",
    "font_color": "#e6edf3",
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
    "accent": "#1f6feb",
    "accent2": "#388bfd",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "colorscale": [
        [0.0, "#0d1117"],
        [0.2, "#1f2937"],
        [0.4, "#1d4ed8"],
        [0.6, "#2563eb"],
        [0.8, "#3b82f6"],
        [1.0, "#93c5fd"],
    ],
    "categorical": [
        "#1f6feb", "#3fb950", "#d29922", "#f85149",
        "#388bfd", "#56d364", "#e3b341", "#ff7b72",
    ],
}


def apply_plotly_theme(fig, title: str = "", height: int = 400):
    """Apply the AutoRAG 360 enterprise dark theme to any Plotly figure."""
    t = PLOTLY_THEME
    layout_updates = dict(
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["bg"],
        font=dict(color=t["font_color"], family=t["font_family"], size=12),
        height=height,
        margin=dict(l=48, r=24, t=48 if title else 24, b=40),
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=t["grid_color"], zeroline=False, linecolor="#30363d"),
        yaxis=dict(gridcolor=t["grid_color"], zeroline=False, linecolor="#30363d"),
    )
    if title:
        layout_updates["title"] = dict(
            text=title,
            font=dict(color=t["font_color"], size=14, family=t["font_family"]),
            x=0.0,
            xanchor="left",
        )
    fig.update_layout(**layout_updates)
    return fig


# ─── Global CSS ───────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ── */
:root {
    --bg-canvas:   #0d1117;
    --bg-default:  #0d1117;
    --bg-subtle:   #161b22;
    --bg-inset:    #010409;
    --border-default: #30363d;
    --border-muted:   #21262d;
    --text-primary:   #e6edf3;
    --text-secondary: #8b949e;
    --text-muted:     #6e7681;
    --text-link:      #58a6ff;
    --accent-fg:      #1f6feb;
    --accent-emphasis: #1f6feb;
    --success-fg:  #3fb950;
    --warning-fg:  #d29922;
    --danger-fg:   #f85149;
    --font-sans:   'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono:   'JetBrains Mono', 'Fira Code', Consolas, monospace;
    --radius-sm:   6px;
    --radius-md:   8px;
    --radius-lg:   12px;
    --shadow-sm:   0 1px 0 rgba(255,255,255,0.03);
    --shadow-md:   0 4px 16px rgba(0,0,0,0.4);
    --transition:  0.15s ease;
}

/* ── Base reset ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: var(--bg-default) !important;
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-inset); }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ═══════════════════════════════════════
   SIDEBAR — Complete overhaul
   ═══════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
    min-width: 260px !important;
}

[data-testid="stSidebar"] > div {
    background: #0d1117 !important;
}

/* Sidebar all text — force white/readable */
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* Sidebar label/caption override */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #c9d1d9 !important;
}

/* Radio group wrapper */
[data-testid="stSidebar"] .stRadio {
    background: transparent !important;
}

/* Radio label text — force fully visible */
[data-testid="stSidebar"] .stRadio > div > div > div > label {
    color: #c9d1d9 !important;
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    padding: 7px 10px 7px 8px !important;
    margin: 1px 0 !important;
    display: flex !important;
    align-items: center !important;
    cursor: pointer !important;
    transition: background var(--transition) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    gap: 6px !important;
}

[data-testid="stSidebar"] .stRadio > div > div > div > label:hover {
    background: rgba(177,186,196,0.1) !important;
    color: #e6edf3 !important;
}

/* Active/selected radio */
[data-testid="stSidebar"] .stRadio > div > div > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > div > div > label[aria-checked="true"] {
    background: rgba(31,111,235,0.15) !important;
    color: #58a6ff !important;
    font-weight: 600 !important;
}

/* Hide the actual radio circle dot — make it pill style */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child {
    display: none !important;
}

/* ═══════════════════════════════════════
   TYPOGRAPHY
   ═══════════════════════════════════════ */
h1 {
    font-size: 1.625rem !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.01em !important;
    line-height: 1.3 !important;
    margin-bottom: 4px !important;
    /* Override any gradient from previous CSS */
    -webkit-text-fill-color: #e6edf3 !important;
    background: none !important;
    background-clip: unset !important;
    -webkit-background-clip: unset !important;
}

h2 {
    font-size: 1.0625rem !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
    border-left: none !important;
    padding-left: 0 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #21262d !important;
    margin-top: 24px !important;
    margin-bottom: 16px !important;
}

h3 {
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    color: #c9d1d9 !important;
}

p, li, span, div {
    color: var(--text-primary);
    font-family: var(--font-sans) !important;
}

/* ═══════════════════════════════════════
   METRIC CARDS
   ═══════════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border-muted) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px 20px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color var(--transition) !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-default) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}

[data-testid="stMetricDelta"] {
    color: var(--text-secondary) !important;
    font-size: 0.8125rem !important;
}

/* ═══════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════ */
.stButton > button {
    background: var(--accent-fg) !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    transition: background var(--transition), box-shadow var(--transition) !important;
    font-family: var(--font-sans) !important;
}

.stButton > button:hover {
    background: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(31,111,235,0.25) !important;
}

.stButton > button:active {
    background: #1a5cc4 !important;
}

/* ═══════════════════════════════════════
   INPUTS
   ═══════════════════════════════════════ */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div {
    background: var(--bg-inset) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.875rem !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent-fg) !important;
    box-shadow: 0 0 0 3px rgba(31,111,235,0.25) !important;
    outline: none !important;
}

[data-baseweb="select"] * {
    color: var(--text-primary) !important;
    background: var(--bg-inset) !important;
}

/* ═══════════════════════════════════════
   DATAFRAMES / TABLES
   ═══════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-muted) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] table {
    background: var(--bg-subtle) !important;
}

/* ═══════════════════════════════════════
   TABS
   ═══════════════════════════════════════ */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border-muted) !important;
    gap: 0 !important;
    padding: 0 !important;
}

[data-testid="stTabs"] [role="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
    background: transparent !important;
    transition: color var(--transition), border-color var(--transition) !important;
}

[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--text-primary) !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: var(--accent-fg) !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* ═══════════════════════════════════════
   ALERTS
   ═══════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 0.875rem !important;
    border-width: 1px !important;
}

/* ═══════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════ */
[data-testid="stProgressBar"] > div > div {
    background: var(--accent-fg) !important;
    border-radius: 4px !important;
}

/* ═══════════════════════════════════════
   CAPTION
   ═══════════════════════════════════════ */
.stCaption, [data-testid="stCaption"], small {
    color: var(--text-muted) !important;
    font-size: 0.8125rem !important;
}

/* ═══════════════════════════════════════
   SLIDER
   ═══════════════════════════════════════ */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--accent-fg) !important;
    border-color: var(--accent-fg) !important;
}

/* ═══════════════════════════════════════
   NUMBER INPUT
   ═══════════════════════════════════════ */
[data-testid="stNumberInput"] input {
    background: var(--bg-inset) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}

/* ═══════════════════════════════════════
   EXPANDER
   ═══════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid var(--border-muted) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-subtle) !important;
}

/* ═══════════════════════════════════════
   HR DIVIDER
   ═══════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid var(--border-muted) !important;
    margin: 24px 0 !important;
}

/* ═══════════════════════════════════════
   SECTION HEADER
   ═══════════════════════════════════════ */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
}

.section-hdr-icon {
    font-size: 0.9rem;
    opacity: 0.7;
}

.section-hdr-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: #c9d1d9;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.8125rem;
}

/* ═══════════════════════════════════════
   KPI / STAT CARDS
   ═══════════════════════════════════════ */
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 18px;
    transition: border-color 0.15s;
}

.kpi-card:hover {
    border-color: #30363d;
}

.kpi-label {
    font-size: 0.6875rem;
    font-weight: 600;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 1.625rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
}

.kpi-sub {
    font-size: 0.75rem;
    color: #6e7681;
    margin-top: 4px;
}

/* ═══════════════════════════════════════
   PAGE HEADER
   ═══════════════════════════════════════ */
.page-header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #21262d;
}

.page-header-tag {
    font-size: 0.6875rem;
    font-weight: 600;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.page-header-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 6px;
    line-height: 1.25;
}

.page-header-desc {
    font-size: 0.875rem;
    color: #8b949e;
    line-height: 1.6;
    max-width: 640px;
}

/* ═══════════════════════════════════════
   CHAT BUBBLES
   ═══════════════════════════════════════ */
.chat-query {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px 8px 2px 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.875rem;
    color: #e6edf3;
    line-height: 1.6;
    max-width: 80%;
    margin-left: auto;
}

.chat-answer {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px 8px 8px 2px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.875rem;
    color: #c9d1d9;
    line-height: 1.7;
}

/* ═══════════════════════════════════════
   SOURCE CITATION CARDS
   ═══════════════════════════════════════ */
.source-item {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.8125rem;
    transition: border-color 0.15s;
}

.source-item:hover {
    border-color: #30363d;
}

/* ═══════════════════════════════════════
   MODE BADGES
   ═══════════════════════════════════════ */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.6875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.badge-blue    { background: rgba(31,111,235,0.15); color: #58a6ff; border: 1px solid rgba(31,111,235,0.3); }
.badge-green   { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
.badge-yellow  { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
.badge-red     { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
.badge-gray    { background: rgba(139,148,158,0.15);color: #8b949e; border: 1px solid rgba(139,148,158,0.3); }

/* Risk level pills */
.risk-high   { display:inline-block; padding:4px 10px; border-radius:4px; font-size:0.8125rem; font-weight:600; background:rgba(248,81,73,0.1); border:1px solid rgba(248,81,73,0.3); color:#f85149; }
.risk-medium { display:inline-block; padding:4px 10px; border-radius:4px; font-size:0.8125rem; font-weight:600; background:rgba(210,153,34,0.1); border:1px solid rgba(210,153,34,0.3); color:#d29922; }
.risk-low    { display:inline-block; padding:4px 10px; border-radius:4px; font-size:0.8125rem; font-weight:600; background:rgba(63,185,80,0.1);  border:1px solid rgba(63,185,80,0.3);  color:#3fb950; }

/* ═══════════════════════════════════════
   SIDEBAR LOGO
   ═══════════════════════════════════════ */
.sb-logo {
    padding: 16px 12px 12px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 8px;
}

.sb-logo-name {
    font-size: 1rem;
    font-weight: 700;
    color: #e6edf3 !important;
    letter-spacing: -0.01em;
}

.sb-logo-sub {
    font-size: 0.6875rem;
    color: #6e7681 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

.sb-status-row {
    padding: 6px 12px;
    font-size: 0.75rem;
    color: #6e7681 !important;
    display: flex;
    align-items: center;
    gap: 6px;
}

.sb-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #3fb950;
    flex-shrink: 0;
}

.sb-section-label {
    padding: 8px 12px 4px;
    font-size: 0.6875rem;
    font-weight: 600;
    color: #6e7681 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ═══════════════════════════════════════
   RESULT / INFO CARDS
   ═══════════════════════════════════════ */
.info-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
}

.info-card-label {
    font-size: 0.6875rem;
    font-weight: 600;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

/* ═══════════════════════════════════════
   PAGE FADE-IN
   ═══════════════════════════════════════ */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stVerticalBlock"] > div:first-child {
    animation: fadeIn 0.2s ease-out;
}

/* ═══════════════════════════════════════
   STREAMLIT SPECIFIC OVERRIDES
   ═══════════════════════════════════════ */
/* Hide default streamlit header decoration */
[data-testid="stDecoration"] { display: none !important; }

/* Block container padding */
[data-testid="block-container"] {
    padding-top: 24px !important;
    padding-bottom: 24px !important;
}

/* Selectbox dropdown */
[data-baseweb="popover"] [role="option"] {
    background: #161b22 !important;
    color: #e6edf3 !important;
    font-size: 0.875rem !important;
}

[data-baseweb="popover"] [role="option"]:hover {
    background: #1f2937 !important;
}

/* Spinner text */
[data-testid="stSpinner"] > div {
    color: #8b949e !important;
    font-size: 0.875rem !important;
}
</style>
"""


def inject_global_css() -> None:
    """Inject the global AutoRAG 360 professional CSS design system."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(tag: str, title: str, description: str = "") -> None:
    """Render a clean page header with optional description."""
    desc_html = f'<div class="page-header-desc">{description}</div>' if description else ""
    st.markdown(
        f"""<div class="page-header">
            <div class="page-header-tag">{tag}</div>
            <div class="page-header-title">{title}</div>
            {desc_html}
        </div>""",
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str) -> None:
    """Render a styled section separator."""
    st.markdown(
        f"""<div class="section-hdr">
            <span class="section-hdr-icon">{icon}</span>
            <span class="section-hdr-title">{title}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", value_color: str = "#e6edf3") -> None:
    """Render a clean, professional KPI card."""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{value_color}">{value}</div>
            {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def badge(text: str, style: str = "gray") -> str:
    """Return an HTML badge."""
    return f'<span class="badge badge-{style}">{text}</span>'
