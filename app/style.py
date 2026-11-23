"""Visual system for the Regional Mobility Change Profiles dashboard.

Presentation only: colours, typography, spacing and the Plotly template.
The stylesheet lives in app/theme.css; this module injects it with the
active theme's colour tokens.

Design direction: an editorial research brief. Warm paper light theme,
slate-blue dark theme, serif display headings, humanist sans body text,
and a restrained petrol accent. All fonts are system fonts, so nothing
depends on a network request.
"""

from pathlib import Path

import plotly.io as pio
import streamlit as st

# System-only faces: Iowan/Palatino/Georgia resolve on macOS and Windows,
# the generic serif on Linux. No webfont is requested or assumed.
DISPLAY_FONT = ('"Iowan Old Style", "Palatino Linotype", Palatino, '
                '"Book Antiqua", Georgia, "Times New Roman", serif')
FONT_STACK = ('"Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, '
              '"Helvetica Neue", Arial, sans-serif')
MONO_FONT = ('"Cascadia Mono", "SF Mono", "Roboto Mono", Consolas, '
             '"Liberation Mono", monospace')

# Analytical cluster palette: petrol, terracotta, indigo, ochre, plum,
# moss, steel and stone. Muted enough for print, distinct in a legend.
CLUSTER_COLORS_LIGHT = [
    "#0F6B62", "#B0563B", "#4C5F9E", "#8A6A1F",
    "#7C4A63", "#4F7A52", "#2F6E8C", "#6B6F76",
]
CLUSTER_COLORS_DARK = [
    "#5FBFAF", "#D98A63", "#8C9BD4", "#C7A75A",
    "#C08CA4", "#86B58A", "#6FB4D0", "#A9B0B8",
]

# Semantic palette for the temporal behavioural states, ordered from the most
# suppressed activity to the most active.
STATE_COLORS_LIGHT = {
    "Suppressed": "#4C5F9E",
    "Near baseline": "#8A8577",
    "Above baseline": "#4F7A52",
    "Well above baseline": "#0F6B62",
}
STATE_COLORS_DARK = {
    "Suppressed": "#8C9BD4",
    "Near baseline": "#B3AD9C",
    "Above baseline": "#86B58A",
    "Well above baseline": "#5FBFAF",
}

SERIES_COLORS_LIGHT = ["#0F6B62", "#4C5F9E", "#B0563B",
                       "#8A6A1F", "#2F6E8C", "#7C4A63"]
SERIES_COLORS_DARK = ["#5FBFAF", "#8C9BD4", "#D98A63",
                      "#C7A75A", "#6FB4D0", "#C08CA4"]

# Diverging scale for standardised values: steel blue to terracotta with a
# neutral midpoint that matches each theme's card surface.
DIVERGING_LIGHT = [[0, "#3E5C76"], [0.25, "#9FB1C0"],
                   [0.5, "#FAF8F3"], [0.75, "#DCA48E"], [1, "#A9502F"]]
DIVERGING_DARK = [[0, "#5E86A8"], [0.25, "#40566B"],
                  [0.5, "#242E3A"], [0.75, "#8A5A48"], [1, "#C77E60"]]

# Sequential scale for probability heatmaps. Light mode keeps Plotly's Blues;
# dark mode runs from the card surface to the readable dark-theme accent so no
# near-white cells appear inside the dark canvas.
SEQUENTIAL_LIGHT = "Blues"
SEQUENTIAL_DARK = [[0, "#242E3A"], [0.25, "#2E474E"],
                   [0.5, "#3E6868"], [0.75, "#579A8E"], [1, "#86D8C8"]]

# Backwards-compatible aliases (dark palette, the historical default).
CLUSTER_COLORS = CLUSTER_COLORS_DARK
SERIES_COLORS = SERIES_COLORS_DARK
STATE_COLORS = STATE_COLORS_DARK

_TOKENS_LIGHT = {
    "bg": "#F3F1EC",
    "surface": "#FFFFFF",
    "surface_alt": "#EBE7DE",
    "sidebar": "#E9E5DC",
    "text": "#1C2430",
    "muted": "#5D6873",
    "accent": "#0F6B62",
    "accent_soft": "rgba(15, 107, 98, 0.08)",
    "accent_border": "rgba(15, 107, 98, 0.30)",
    "border": "#DCD6C9",
    "grid": "#E6E1D6",
    "plot": "#FFFFFF",
    "hover": "#FFFFFF",
    "scheme": "light",
    "table_filter": "none",
}

_TOKENS_DARK = {
    "bg": "#1B232D",
    "surface": "#242E3A",
    "surface_alt": "#2C3743",
    "sidebar": "#161D26",
    "text": "#E9EEF3",
    "muted": "#A7B4C2",
    "accent": "#6FC7B8",
    "accent_soft": "rgba(111, 199, 184, 0.10)",
    "accent_border": "rgba(111, 199, 184, 0.38)",
    "border": "#3A4756",
    "grid": "#333F4C",
    "plot": "#242E3A",
    "hover": "#2C3743",
    "scheme": "dark",
    # Streamlit renders dataframe headers and cells on a canvas, so they do not
    # inherit the injected CSS. This scoped filter is the only non-destructive
    # way to darken both canvases while keeping sorting, scrolling and values.
    "table_filter": ("invert(0.88) sepia(0.75) hue-rotate(180deg) "
                     "saturate(2.4) brightness(1.05)"),
}

_MAP_COLORS = {
    "light": {"land": "#E7E2D6", "ocean": "#F3F1EC",
              "boundary": "#C9C1B1", "no_data": "#DDD7C9"},
    "dark": {"land": "#222C37", "ocean": "#1B232D",
             "boundary": "#46535F", "no_data": "#333E4A"},
}

_CSS_PATH = Path(__file__).with_name("theme.css")


def palette(theme="Light"):
    """Return the colour sets for the requested appearance."""
    light = theme != "Dark"
    return {
        "cluster": CLUSTER_COLORS_LIGHT if light else CLUSTER_COLORS_DARK,
        "state": STATE_COLORS_LIGHT if light else STATE_COLORS_DARK,
        "series": SERIES_COLORS_LIGHT if light else SERIES_COLORS_DARK,
    }


def _build_template(colors, light):
    from copy import deepcopy

    template = deepcopy(pio.templates["plotly_white" if light else "plotly_dark"])
    template.layout.update(
        paper_bgcolor=colors["surface"],
        plot_bgcolor=colors["surface"],
        font=dict(family=FONT_STACK, color=colors["text"], size=12),
        title=dict(font=dict(size=13, color=colors["text"]), x=0,
                   xanchor="left", pad=dict(b=2)),
        xaxis=dict(
            gridcolor=colors["grid"],
            zerolinecolor=colors["border"],
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(color=colors["muted"], size=11),
            title_font=dict(color=colors["muted"], size=11.5),
            automargin=True,
            ticks="",
        ),
        yaxis=dict(
            gridcolor=colors["grid"],
            zerolinecolor=colors["border"],
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(color=colors["muted"], size=11),
            title_font=dict(color=colors["muted"], size=11.5),
            automargin=True,
            ticks="",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    font=dict(color=colors["muted"], size=11.5),
                    itemsizing="constant"),
        hoverlabel=dict(bgcolor=colors["hover"], bordercolor=colors["border"],
                        font=dict(color=colors["text"], size=12,
                                  family=FONT_STACK)),
        margin=dict(l=14, r=14, t=44, b=36),
        colorway=palette("Light" if light else "Dark")["series"],
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return template


def apply_dashboard_style(theme="Light"):
    """Apply the visual system to Streamlit and Plotly.

    Returns the chart-surface colours so map and chart backgrounds come from
    the same tokens as the interface.
    """
    light = theme != "Dark"
    colors = _TOKENS_LIGHT if light else _TOKENS_DARK
    maps = _MAP_COLORS["light" if light else "dark"]

    template_name = "mobility_light" if light else "mobility_dark"
    pio.templates[template_name] = _build_template(colors, light)
    pio.templates.default = template_name

    css = _CSS_PATH.read_text(encoding="utf-8")
    replacements = {
        "__BG__": colors["bg"],
        "__SURFACE__": colors["surface"],
        "__ALT__": colors["surface_alt"],
        "__SIDEBAR__": colors["sidebar"],
        "__TEXT__": colors["text"],
        "__MUTED__": colors["muted"],
        "__ACCENT_SOFT__": colors["accent_soft"],
        "__ACCENT_BORDER__": colors["accent_border"],
        "__ACCENT__": colors["accent"],
        "__BORDER__": colors["border"],
        "__PLOT__": colors["plot"],
        "__FONT__": FONT_STACK,
        "__DISPLAY_FONT__": DISPLAY_FONT,
        "__MONO_FONT__": MONO_FONT,
        "__SCHEME__": colors["scheme"],
        "__TABLE_FILTER__": colors["table_filter"],
    }
    for key, value in replacements.items():
        css = css.replace(key, value)

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    return {
        "surface": colors["surface"],
        "text": colors["text"],
        "muted": colors["muted"],
        "grid": colors["grid"],
        "border": colors["border"],
        "accent": colors["accent"],
        "diverging": DIVERGING_LIGHT if light else DIVERGING_DARK,
        "sequential": SEQUENTIAL_LIGHT if light else SEQUENTIAL_DARK,
        **maps,
    }
