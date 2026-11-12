
"""Shared visual theme for the mobility research dashboard."""

from copy import deepcopy

import plotly.io as pio
import streamlit as st


# ------------------------------------------------------------
# Design tokens
# ------------------------------------------------------------

BACKGROUND = "#0B1220"
SURFACE = "#142235"
SURFACE_ALT = "#18283A"
TEXT = "#E8EEF5"
MUTED = "#A8B6C7"
ACCENT = "#55C6B3"
BORDER = "rgba(148, 172, 196, 0.18)"


# Colour-blind-conscious palette for analytical clusters.
CLUSTER_COLORS = [
    "#55C6B3",  # Cluster 0 ? Teal
    "#E6A23C",  # Cluster 1 ? Warm amber
    "#6FA8DC",  # Cluster 2 ? Muted blue
    "#CC79A7",  # Cluster 3 ? Soft rose
    "#009E73",  # Cluster 4 ? Green
    "#A8B6C7",  # Cluster 5 ? Slate
    "#8DA0CB",  # Cluster 6 ? Periwinkle
    "#D5A6BD",  # Cluster 7 ? Dusty pink
]


# Separate semantic palette for temporal behavioural states.
STATE_COLORS = {
    "Suppressed": "#56B4E9",
    "Near baseline": "#A8B6C7",
    "Above baseline": "#E69F00",
    "Well above baseline": "#009E73",
}


# General analytical series colours.
SERIES_COLORS = [
    "#55C6B3",
    "#56B4E9",
    "#E69F00",
    "#CC79A7",
    "#D55E00",
    "#009E73",
]


# ------------------------------------------------------------
# Shared Plotly theme
# ------------------------------------------------------------

# Make an independent copy rather than modifying Plotly's
# built-in template.
mobility_template = deepcopy(pio.templates["plotly_dark"])

mobility_template.layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Inter, Segoe UI, sans-serif",
        color=TEXT,
        size=13,
    ),
    title=dict(
        font=dict(size=18, color=TEXT),
        x=0,
        xanchor="left",
    ),
    xaxis=dict(
        gridcolor="rgba(168,182,199,0.14)",
        zerolinecolor="rgba(168,182,199,0.25)",
        linecolor="rgba(168,182,199,0.20)",
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT),
    ),
    yaxis=dict(
        gridcolor="rgba(168,182,199,0.14)",
        zerolinecolor="rgba(168,182,199,0.25)",
        linecolor="rgba(168,182,199,0.20)",
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
    ),
    colorway=SERIES_COLORS,
    margin=dict(l=24, r=24, t=55, b=40),
    hoverlabel=dict(
        bgcolor=SURFACE_ALT,
        bordercolor=BORDER,
        font=dict(color=TEXT),
    ),
    geo=dict(
        bgcolor="rgba(0,0,0,0)",
    ),
)

pio.templates["mobility_dark"] = mobility_template
pio.templates.default = "mobility_dark"


# ------------------------------------------------------------
# Streamlit interface styling
# ------------------------------------------------------------

def apply_dashboard_style():
    """Apply the shared research dashboard styling."""

    st.markdown(
        """
        <style>
        :root {
            --mobility-bg: #0B1220;
            --mobility-surface: #142235;
            --mobility-surface-alt: #18283A;
            --mobility-border: rgba(148, 172, 196, 0.18);
            --mobility-text: #E8EEF5;
            --mobility-muted: #A8B6C7;
            --mobility-accent: #55C6B3;
        }

        /* Main application background */
        .stApp {
            background:
                radial-gradient(
                    ellipse at 8% 0%,
                    rgba(0, 114, 178, 0.16),
                    transparent 38%
                ),
                radial-gradient(
                    ellipse at 95% 8%,
                    rgba(0, 158, 115, 0.12),
                    transparent 34%
                ),
                linear-gradient(
                    145deg,
                    #0B1220 0%,
                    #101B2B 55%,
                    #0B1722 100%
                );
            color: var(--mobility-text);
        }

        /* Balanced research dashboard layout */
        [data-testid="stMainBlockContainer"] {
            max-width: 1380px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stMainBlockContainer"] > div {
            gap: 1.15rem;
        }

        /* Header */
        [data-testid="stHeader"] {
            background: rgba(11, 18, 32, 0.78);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #101C2D 0%,
                #0D1726 100%
            );
            border-right: 1px solid var(--mobility-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        /* Professional typography hierarchy */
        .stApp h1 {
            color: var(--mobility-text) !important;
            font-size: 1.9rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.035em !important;
        }

        .stApp h2 {
            color: var(--mobility-text) !important;
            font-size: 1.35rem !important;
            font-weight: 650 !important;
            letter-spacing: -0.02em;
        }

        .stApp h3 {
            color: var(--mobility-text) !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
        }

        .stApp p,
        .stApp label,
        .stApp [data-testid="stCaptionContainer"] {
            color: var(--mobility-muted);
            line-height: 1.6;
        }

        .stApp [data-testid="stCaptionContainer"] {
            font-size: 0.88rem;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: rgba(20, 34, 53, 0.88);
            border: 1px solid var(--mobility-border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 26px rgba(0, 0, 0, 0.14);
        }

        [data-testid="stMetricLabel"] {
            color: var(--mobility-muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--mobility-text);
        }

        /* Bordered containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--mobility-border);
            border-radius: 16px;
            background: rgba(20, 34, 53, 0.62);
        }

        /* Tabs */
        [data-testid="stTabs"] button {
            font-weight: 600;
            color: var(--mobility-muted);
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--mobility-accent);
        }

        /* Charts */
        [data-testid="stPlotlyChart"] {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
        }

        /* Tables */
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--mobility-border);
            border-radius: 12px;
            overflow: hidden;
        }

        /* Alerts and links */
        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        a {
            color: var(--mobility-accent);
        }

        /* Research Intelligence teal accent */
        :root {
            --primary-color: #55C6B3;
            accent-color: #55C6B3;
        }

        /* Active tab indicator and text */
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #55C6B3 !important;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #55C6B3 !important;
        }

        /* Radio buttons */
        [data-testid="stRadio"] label[data-checked="true"] > div:first-child {
            border-color: #55C6B3 !important;
            background-color: #55C6B3 !important;
        }

        /* Slider track and thumb */
        [data-testid="stSlider"] [role="slider"] {
            background-color: #55C6B3 !important;
            border-color: #55C6B3 !important;
        }

        [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background-color: #55C6B3 !important;
        }

        /* Focused input controls */
        [data-baseweb="select"]:focus-within,
        [data-baseweb="input"]:focus-within,
        [data-baseweb="textarea"]:focus-within {
            border-color: #55C6B3 !important;
        }

        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
