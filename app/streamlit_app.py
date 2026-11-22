"""Dashboard:  python -m streamlit run app/streamlit_app.py  (run from project root)"""
import json
import os
import sys
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from src.cluster import SCALERS, fit_euclidean, kmedians, load_config, load_features
from app.style import apply_dashboard_style, palette

CFG = load_config()
PROCESSED = Path(CFG["paths"]["processed"])
TABLES = Path(CFG["paths"]["tables"])
SEED = CFG["project"]["random_seed"]
N_INIT = CFG["clustering"]["n_init"]
PERIODS = CFG["periods"]
CATEGORIES = ["retail_and_recreation", "grocery_and_pharmacy", "parks",
              "transit_stations", "workplaces", "residential"]
SCOPES = {"Countries (national)": "national",
          "India states": "india_states",
          "India districts": "india_districts", "Global states": "global_states"}
FEATURE_LABELS = {"no_parks": "Parks excluded", "full": "All six categories"}
TAB_LABELS = ["Overview", "Profiles", "Geography", "Validation",
              "Timeline", "Compare", "Method"]

st.set_page_config(page_title="Regional mobility change profiles", layout="wide")


@st.cache_data(show_spinner=False)
def get_features(scope, feature_set):
    return load_features(PROCESSED / f"{scope}_features_region.parquet", feature_set)


@st.cache_data(show_spinner="Clustering regions...")
def run_clustering(scope, feature_set, scaler, distance, k):
    """Same scaler, algorithm, seed and trough-depth relabelling as before."""
    base = get_features(scope, feature_set)
    X = SCALERS[scaler]().fit_transform(base.to_numpy())
    if distance == "euclidean":
        labels = fit_euclidean(X, k, N_INIT, SEED)[0]
    else:
        labels = kmedians(X, k, N_INIT, SEED)[0]
    order = (base.assign(c=labels).groupby("c")["trough_depth"].mean()
             .sort_values(ascending=False).index)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[v] for v in labels])
    pca = PCA(n_components=2).fit(X)
    return labels, pca.transform(X), pca.explained_variance_ratio_


@st.cache_data(show_spinner=False)
def get_daily(scope):
    return pd.read_parquet(PROCESSED / f"{scope}_daily_clean.parquet")


@st.cache_data(show_spinner=False)
def read_table(name):
    path = TABLES / name
    return pd.read_csv(path, keep_default_na=False) if path.exists() else None


def first_match(pattern):
    files = sorted(TABLES.glob(pattern))
    return files[0] if files else None


def _shift_page(page_key, delta, total_pages):
    """Move the membership table by one page (button callback)."""
    current = int(st.session_state.get(page_key, 1))
    st.session_state[page_key] = max(1, min(total_pages, current + delta))


@st.cache_data(show_spinner=False)
def country_iso3():
    """Map country names to ISO-3 codes for the choropleth (None if pycountry is missing)."""
    try:
        import pycountry
    except ImportError:
        return None
    meta = pd.read_csv(PROCESSED / "national_region_meta.csv", keep_default_na=False)
    codes = {}
    for name, code in zip(meta["country_region"], meta["country_region_code"]):
        try:
            found = pycountry.countries.get(alpha_2=code)
        except (KeyError, LookupError):
            found = None
        if found is not None:
            codes[name] = found.alpha_3
    return codes


def describe_clusters(base, labels):
    """Plain-language name and description for each cluster, computed from its profile.
    Clusters are numbered from the mildest first-wave dip (0) to the deepest."""
    prof = base.assign(cluster=labels).groupby("cluster").mean()
    z = (prof - base.mean()) / base.std()
    k_ = len(prof)
    sizes = np.bincount(labels, minlength=k_)
    late_cols = [c for c in base.columns
                 if c.endswith("__late_mean") and not c.startswith("residential")]
    out = {}
    for c in prof.index:
        if c == 0:
            dip = "Mildest dip"
        elif c == k_ - 1:
            dip = "Deepest dip"
        elif k_ == 3:
            dip = "Intermediate dip"
        else:
            dip = f"Dip level {c + 1} of {k_}"
        rec_z = z.loc[c, "recovery_days"]
        if rec_z > 0.5:
            recovery = "slow recovery"
        elif rec_z < -0.5:
            recovery = "fast recovery"
        else:
            recovery = "average recovery"
        name = f"Cluster {c}: {dip}, {recovery}"
        text = (f"{sizes[c]} regions. In the first wave, activity fell to about "
                f"{abs(prof.loc[c, 'trough_depth']):.0f}% below baseline at its lowest, "
                f"and took on average {prof.loc[c, 'recovery_days']:.0f} days to return to "
                f"within 10 points of baseline. In the late period, retail averaged "
                f"{prof.loc[c, 'retail_and_recreation__late_mean']:+.0f}% and workplaces "
                f"{prof.loc[c, 'workplaces__late_mean']:+.0f}% versus the Jan-Feb 2020 reference.")
        if z.loc[c, late_cols].mean() > 0.5:
            text += " Late-period activity is well above the typical region."
        out[c] = (name, text)
    return out


def state_labels(scope):
    """Names for the behavioural states, based on each state's average activity."""
    f = first_match(f"state_profiles_{scope}_k*.csv")
    if f is None:
        return {}
    sp = pd.read_csv(f, index_col=0)
    act = sp[["retail_and_recreation__mean", "transit_stations__mean",
              "workplaces__mean"]].mean(axis=1)

    def label(a):
        if a < -20:
            return "Suppressed"
        if a < 8:
            return "Near baseline"
        if a < 45:
            return "Above baseline"
        return "Well above baseline"

    return {int(s): f"State {int(s)}: {label(a)}" for s, a in act.items()}


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(
        '<div class="mob-side-brand">Mobility change profiles</div>'
        '<div class="mob-side-kicker">Exploratory clustering of regional '
        'activity during COVID-19.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mob-side-label">Analysis</div>', unsafe_allow_html=True)
    scope_label = st.selectbox("Region level", list(SCOPES))
    scope = SCOPES[scope_label]
    feature_set = st.radio(
        "Feature set", ["no_parks", "full"],
        format_func=lambda value: FEATURE_LABELS[value],
        help="Parks is seasonal (baseline is Jan-Feb 2020), so it is excluded by default.",
    )
    scaler = st.radio(
        "Scaler", ["robust", "standard"],
        format_func=lambda value: value.title() + " scaling",
    )
    distance = st.radio(
        "Algorithm", ["euclidean", "manhattan"],
        format_func=lambda value: ("K-Means (Euclidean)" if value == "euclidean"
                                   else "K-Medians (Manhattan)"),
    )
    k = st.slider("Number of clusters (k)", 2, 8, 3)

    st.markdown('<div class="mob-side-label">Display</div>', unsafe_allow_html=True)
    theme = st.radio("Appearance", ["Light", "Dark"], horizontal=True,
                     key="appearance_theme")

    st.markdown(
        f'<div class="mob-side-foot">Google Community Mobility Reports<br>'
        f'February 2020 – October 2022</div>',
        unsafe_allow_html=True,
    )

chart = apply_dashboard_style(theme)
pal = palette(theme)
method_label = "K-Means" if distance == "euclidean" else "K-Medians"
distance_label = "Euclidean" if distance == "euclidean" else "Manhattan"

base = get_features(scope, feature_set)
labels, Z, evr = run_clustering(scope, feature_set, scaler, distance, k)
desc = describe_clusters(base, labels)
cluster_order = [desc[i][0] for i in range(k)]

# ------------------------------------------------------------------- header
st.markdown(
    f"""
    <div class="mob-masthead">
        <div class="mob-kicker">Mobility research · COVID-19</div>
        <h1 class="mob-title">Regional Mobility Change Profiles</h1>
        <p class="mob-lede">
            How public activity shifted across regions during COVID-19, which
            regions changed in similar ways, and how well those groupings hold up.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_profiles, tab_geography, tab_evidence, tab_time, tab_compare, tab_method = st.tabs(TAB_LABELS)
# ------------------------------------------------------------------ overview
with tab_overview:
    cluster_counts = np.bincount(labels, minlength=k)
    largest_cluster = int(np.argmax(cluster_counts))
    largest_count = int(cluster_counts[largest_cluster])
    largest_name, largest_description = desc[largest_cluster]
    largest_share = largest_count / len(labels) if len(labels) else 0

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">01</span>'
        'At a glance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="mob-kpis">
            <div class="mob-kpi">
                <div class="mob-kpi-label">Regions analysed</div>
                <div class="mob-kpi-value">{len(base):,}</div>
                <div class="mob-kpi-note">Complete mobility profiles at this level</div>
            </div>
            <div class="mob-kpi">
                <div class="mob-kpi-label">Distinct profiles</div>
                <div class="mob-kpi-value">{k}</div>
                <div class="mob-kpi-note">{method_label} - {scaler.title()} scaling</div>
            </div>
            <div class="mob-kpi">
                <div class="mob-kpi-label">Variance shown in two dimensions</div>
                <div class="mob-kpi-value">{evr.sum():.1%}</div>
                <div class="mob-kpi-note">PC1 {evr[0]:.1%} - PC2 {evr[1]:.1%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">02</span>'
        'Research framing</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mob-grid-2">
            <div class="mob-card">
                <div class="mob-card-title">Research question</div>
                <p class="mob-card-text">
                    Can regions be grouped into distinct mobility-change profiles
                    based on how public activity changed during the COVID-19
                    pandemic?
                </p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Data and method</div>
                <p class="mob-card-text">
                    Google COVID-19 Community Mobility Reports, February 2020 to
                    October 2022, analysed at national, Indian state, Indian district
                    and global state levels.
                </p>
                <p class="mob-card-text">
                    Regional profiles are scaled, clustered with K-Means or
                    K-Medians, projected with PCA and checked with stability and
                    hold-out validation.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">03</span>'
        'Research workflow</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mob-flow">
            <div class="mob-step">
                <span class="mob-step-num">01</span>
                <div class="mob-step-title">Data preparation</div>
                <div class="mob-step-text">Cleaning, gap filling and feature selection</div>
            </div>
            <div class="mob-step">
                <span class="mob-step-num">02</span>
                <div class="mob-step-title">Feature scaling</div>
                <div class="mob-step-text">Robust or standard scaling of regional profiles</div>
            </div>
            <div class="mob-step">
                <span class="mob-step-num">03</span>
                <div class="mob-step-title">Clustering</div>
                <div class="mob-step-text">K-Means or K-Medians, ordered by trough depth</div>
            </div>
            <div class="mob-step">
                <span class="mob-step-num">04</span>
                <div class="mob-step-title">Validation</div>
                <div class="mob-step-text">Separation, stability and out-of-sample checks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">04</span>'
        'What this run shows</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="mob-insights">
            <div class="mob-card">
                <div class="mob-kpi-label">Largest profile - {largest_share:.1%} of regions</div>
                <div class="mob-insight-value">{largest_count}</div>
                <div class="mob-card-title">{largest_name}</div>
                <p class="mob-card-text">{largest_description}</p>
            </div>
            <div class="mob-card">
                <div class="mob-kpi-label">Reading the feature space</div>
                <div class="mob-insight-value">{evr.sum():.1%}</div>
                <div class="mob-card-title">Variance kept by the two PCA axes</div>
                <p class="mob-card-text">
                    PC1 retains {evr[0]:.1%} of the variation between regional
                    profiles and PC2 retains {evr[1]:.1%}. Points that sit close
                    together changed in similar ways; separation between colours
                    shows how distinct the profiles are.
                </p>
                <p class="mob-card-text">
                    Clusters are exploratory groupings, not official categories.
                    The largest cluster is the most common mobility-change profile
                    in the selected dataset.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mob-meta-line">'
        '<span>Configuration</span>'
        f'<span><b>{scope_label}</b></span>'
        f'<span><b>{FEATURE_LABELS[feature_set]}</b></span>'
        f'<span><b>{scaler.title()} scaling</b></span>'
        f'<span><b>{distance_label} distance</b></span>'
        f'<span><b>{method_label}</b></span>'
        f'<span><b>k = {k}</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------- cluster profiles
with tab_profiles:
    sizes = np.bincount(labels, minlength=k)
    prof_means = (base.assign(cluster=labels)
                  .groupby("cluster")[["trough_depth", "recovery_days"]].mean())

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">01</span>'
        'Cluster overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="mob-kpis">
            <div class="mob-kpi">
                <div class="mob-kpi-label">Regions clustered</div>
                <div class="mob-kpi-value">{len(base):,}</div>
                <div class="mob-kpi-note">Each region carries one profile</div>
            </div>
            <div class="mob-kpi">
                <div class="mob-kpi-label">Variance shown by PCA</div>
                <div class="mob-kpi-value">{evr.sum():.0%}</div>
                <div class="mob-kpi-note">PC1 and PC2 combined</div>
            </div>
            <div class="mob-kpi">
                <div class="mob-kpi-label">Smallest profile</div>
                <div class="mob-kpi-value">{int(sizes.min())}</div>
                <div class="mob-kpi-note">Regions in the least common cluster</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What each cluster means", expanded=False):
        for c in range(k):
            name, text = desc[c]
            st.markdown(f"**{name}**")
            st.write(text)

        st.caption("Clusters are numbered from the mildest first-wave dip (0) to the deepest. "
                   "Names are descriptive labels generated from each cluster's averages, not "
                   "official categories. Recovery days for regions that never recovered are "
                   "counted up to the end of the data, so averages are lower bounds. The data "
                   "shows how behaviour changed, not why.")

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">02</span>'
        'Feature space</div>',
        unsafe_allow_html=True,
    )
    chart_col, legend_col = st.columns([1.6, 1], gap="large")

    with chart_col:
        st.markdown('<div class="mob-card-title">Cluster separation</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Each point is one region, projected onto the two principal components of the "
            "scaled profile features. Colours follow the cluster profiles; crosses mark "
            "cluster means."
        )
        plot_df = pd.DataFrame({"region": base.index, "PC1": Z[:, 0], "PC2": Z[:, 1],
                                "cluster": [desc[c][0] for c in labels]})
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster",
                         hover_name="region",
                         color_discrete_sequence=pal["cluster"],
                         category_orders={"cluster": cluster_order},
                         labels={"PC1": f"PC1 ({evr[0]:.0%})",
                                 "PC2": f"PC2 ({evr[1]:.0%})"})
        fig.update_traces(
            marker=dict(size=8.5, opacity=0.92,
                        line=dict(width=0.4, color=chart["border"]))
        )

        centres = (plot_df.groupby("cluster", sort=False)[["PC1", "PC2"]]
                   .mean().reset_index())
        fig.add_scatter(
            x=centres["PC1"],
            y=centres["PC2"],
            mode="markers+text",
            text=centres["cluster"].str.extract(r"Cluster (\d+)")[0].apply(
                lambda value: f"Mean {value}"
            ),
            textposition="top center",
            textfont=dict(color=chart["text"], size=9.5),
            marker=dict(symbol="x", size=12, color=chart["text"],
                        line=dict(width=2)),
            name="Cluster means",
            hovertemplate=("<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}"
                           "<extra></extra>"),
        )
        fig.update_layout(
            height=480,
            margin=dict(l=6, r=6, t=10, b=60),
            paper_bgcolor=chart["surface"],
            plot_bgcolor=chart["surface"],
            hovermode="closest",
            showlegend=False,
        )
        fig.update_xaxes(title_standoff=6)
        fig.update_yaxes(title_standoff=6)
        st.plotly_chart(fig, width="stretch", theme=None)
        st.markdown(
            '<p class="mob-chart-note">Points that sit close together changed in similar '
            'ways. Separation between colours shows how distinct the profiles are.</p>',
            unsafe_allow_html=True,
        )

    with legend_col:
        rows = "".join(
            f'<div class="mob-legend-row">'
            f'<span class="mob-dot" style="background:{pal["cluster"][c]}"></span>'
            f'<div><div class="mob-legend-name">{desc[c][0]}</div>'
            f'<div class="mob-legend-meta">{int(sizes[c])} regions · trough '
            f'{prof_means.loc[c, "trough_depth"]:.0f}% · '
            f'{prof_means.loc[c, "recovery_days"]:.0f} days to recover</div></div>'
            f'</div>'
            for c in range(k)
        )
        st.markdown(
            f'<div class="mob-card">'
            f'<div class="mob-card-title">Cluster identities</div>'
            f'<div class="mob-legend">{rows}</div>'
            f'<p class="mob-card-text" style="margin-top:0.8rem">'
            f'Colours are held constant across the scatter plot, maps, tables and '
            f'temporal charts.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">03</span>'
        'Profile table</div>',
        unsafe_allow_html=True,
    )
    profile = base.assign(cluster=labels).groupby("cluster").mean().round(1)
    profile.insert(0, "n_regions", np.bincount(labels))
    profile.index = [f"Cluster {c}" for c in profile.index]

    display_profile = profile.T.copy()
    metric_labels = {
        "first_wave_mean": "First-wave average change (%)",
        "late_mean": "Late-period average change (%)",
        "trough_depth": "First-wave trough (%)",
        "recovery_days": "Recovery time (days)",
    }
    readable_labels = []
    for feature in display_profile.index:
        if feature == "n_regions":
            readable_labels.append("Number of regions")
        elif "__" in feature:
            category, metric = feature.rsplit("__", 1)
            category = category.replace("_", " ").title()
            metric = metric_labels.get(metric, metric.replace("_", " ").title())
            readable_labels.append(f"{category} — {metric}")
        else:
            readable_labels.append(feature.replace("_", " ").title())

    display_profile.index = readable_labels
    display_profile.index.name = "Feature"

    st.caption("Mean feature values within each cluster. "
               "Number of regions indicates the size of each cluster.")
    profile_styler = display_profile.style.format("{:.1f}")
    profile_styler = profile_styler.format(
        "{:.0f}", subset=(["Number of regions"], slice(None))
    )
    st.dataframe(profile_styler, width="stretch")

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">04</span>'
        'Cluster signatures</div>',
        unsafe_allow_html=True,
    )
    heat_col, guide_col = st.columns([1.85, 1], gap="large")

    z = (base.assign(cluster=labels).groupby("cluster").mean() - base.mean()) / base.std()
    z.index = [f"Cluster {c}" for c in z.index]

    heat_data = z.T.round(2)
    heat_data.index = display_profile.index[1:]

    short_categories = {
        "Retail And Recreation": "Retail",
        "Grocery And Pharmacy": "Grocery",
        "Transit Stations": "Transit",
        "Workplaces": "Workplaces",
        "Residential": "Residential",
    }
    short_metrics = {
        "First-wave average change (%)": "first-wave avg",
        "Late-period average change (%)": "late avg",
        "First-wave trough (%)": "trough",
        "Recovery time (days)": "recovery days",
    }

    short_plain = {"Trough Depth": "Trough", "Recovery Days": "Recovery days"}

    def short_feature(label):
        if " — " not in label:
            return short_plain.get(label, label)
        category, metric = label.split(" — ", 1)
        return (f"{short_categories.get(category, category)} · "
                f"{short_metrics.get(metric, metric)}")

    heat_display = heat_data.copy()
    heat_display.index = [short_feature(label) for label in heat_display.index]
    heat_height = max(360, min(430, len(heat_display) * 26 + 70))

    with heat_col:
        st.markdown('<div class="mob-card-title">Standardised profile features</div>',
                    unsafe_allow_html=True)
        st.caption("Red indicates above-average feature values; blue indicates below-average "
                   "values. Numbers show standardised z-scores, capped for colour at ±2.5.")
        heat = px.imshow(
            heat_display,
            color_continuous_scale=chart["diverging"],
            zmin=-2.5,
            zmax=2.5,
            text_auto=".2f",
            aspect="auto",
            labels=dict(x="Cluster", y="Feature", color="z-score"),
        )
        heat.update_layout(
            paper_bgcolor=chart["surface"],
            plot_bgcolor=chart["surface"],
            margin=dict(l=10, r=60, t=12, b=54),
            height=heat_height,
            coloraxis_colorbar=dict(title="z-score", thickness=11, len=0.8),
        )
        heat.update_xaxes(title_text="Cluster", side="bottom", tickfont=dict(size=11))
        heat.update_yaxes(title_text="", tickfont=dict(size=10.5), automargin=True)
        st.plotly_chart(heat, width="stretch", theme=None)

    with guide_col:
        st.markdown(
            '<div class="mob-card">'
            '<div class="mob-card-title">Reading the feature names</div>'
            '<div class="mob-def">'
            '<div><div class="mob-def-term">First-wave average</div>'
            '<div class="mob-def-text">Average change from the Jan–Feb 2020 baseline '
            'during the first wave (Mar–Jun 2020).</div></div>'
            '<div><div class="mob-def-term">Late-period average</div>'
            '<div class="mob-def-text">Average change during the late pandemic period '
            '(Apr–Oct 2022).</div></div>'
            '<div><div class="mob-def-term">First-wave trough</div>'
            '<div class="mob-def-text">The lowest point of overall activity (retail, '
            'transit, workplaces) in the first wave. More negative means a deeper dip.'
            '</div></div>'
            '<div><div class="mob-def-term">Recovery time</div>'
            '<div class="mob-def-text">Days from that lowest point until activity was '
            'back within 10 points of baseline. Longer means slower recovery.</div></div>'
            '<div><div class="mob-def-term">Residential</div>'
            '<div class="mob-def-text">Time spent at home. It moves in the opposite '
            'direction to the other categories.</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">05</span>'
        'Cluster membership</div>',
        unsafe_allow_html=True,
    )
    st.caption("Search, filter and page through the regions assigned to each profile. "
               "The full assignment is available as CSV.")

    members = pd.DataFrame(
        {"region": base.index, "cluster": labels}
    ).sort_values(["cluster", "region"])
    member_table = members.assign(profile=[desc[c][0] for c in members["cluster"]])
    member_table = member_table.rename(
        columns={"region": "Region", "cluster": "Cluster", "profile": "Cluster profile"}
    )

    with st.container(key="card-membership"):
        filters = st.columns([2, 2.4, 0.9], gap="medium")
        query = filters[0].text_input(
            "Search regions", placeholder="Type part of a region name",
            key=f"member_search_{scope}",
        )
        selected = filters[1].multiselect(
            "Cluster profiles", cluster_order, default=cluster_order,
            format_func=lambda name: name.split(":", 1)[0],
            key=f"member_filter_{scope}_{k}",
        )
        page_size = filters[2].selectbox(
            "Rows per page", [10, 25, 50, 100], index=1,
            key=f"member_size_{scope}",
        )

        view = member_table[member_table["Cluster profile"].isin(selected)]
        if query.strip():
            view = view[view["Region"].str.contains(
                query.strip(), case=False, na=False, regex=False
            )]

        pages = max(1, ceil(len(view) / page_size))
        page_key = f"member_page_{scope}_{k}"
        st.session_state.setdefault(page_key, 1)
        page = max(1, min(int(st.session_state[page_key]), pages))
        st.session_state[page_key] = page

        nav_prev, nav_next, nav_status = st.columns([1, 1, 4], gap="small")
        nav_prev.button(
            "Previous", key=f"member_prev_{scope}_{k}", disabled=page <= 1,
            on_click=_shift_page, args=(page_key, -1, pages),
        )
        nav_next.button(
            "Next", key=f"member_next_{scope}_{k}", disabled=page >= pages,
            on_click=_shift_page, args=(page_key, 1, pages),
        )

        start = (page - 1) * page_size
        page_view = view.iloc[start:start + page_size]
        st.dataframe(page_view, hide_index=True, width="stretch")

        if len(view):
            hint = (f"Page {page} of {pages} · showing {start + 1}–"
                    f"{min(start + page_size, len(view))} of {len(view)} regions")
        else:
            hint = "No regions match the current search and filter."
        nav_status.markdown(f'<div class="mob-hint">{hint}</div>',
                            unsafe_allow_html=True)

    st.download_button(
        "Download assignments (CSV)",
        members.to_csv(index=False),
        file_name=f"clusters_{scope}_k{k}.csv",
        mime="text/csv",
    )
# ------------------------------------------------------------------ geography
with tab_geography:
    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">01</span>'
        'Where profiles sit</div>',
        unsafe_allow_html=True,
    )

    if scope == "india_states":
        boundary_path = ROOT / "data" / "boundaries" / "india_full.geojson"

        if not boundary_path.exists():
            st.warning("India state boundary file is missing.")
        else:
            with boundary_path.open(encoding="utf-8") as f:
                india_geojson = json.load(f)

            boundary_names = [
                feature["properties"]["name"]
                for feature in india_geojson["features"]
            ]
            state_clusters = {
                ("Daman & Diu" if str(region).split(" | ", 1)[-1] == "Daman and Diu"
                 else str(region).split(" | ", 1)[-1]): desc[c][0]
                for region, c in zip(base.index, labels)
            }

            map_df = pd.DataFrame({"map_name": boundary_names})
            map_df["cluster"] = (
                map_df["map_name"].map(state_clusters).fillna("No data")
            )

            state_color_map = {
                desc[i][0]: pal["cluster"][i] for i in range(len(cluster_order))
            }
            state_color_map["No data"] = chart["no_data"]

            with st.container(key="card-geo-controls"):
                controls = st.columns([2.6, 1], gap="large")
                controls[0].markdown(
                    '<div class="mob-card-title">Indian state clusters</div>'
                    '<p class="mob-card-text">Boundary-matched state and union '
                    'territory outlines.</p>',
                    unsafe_allow_html=True,
                )
                state_map_height = controls[1].slider(
                    "Map height", 320, 600, 380, 40, key="india_state_map_height"
                )

            map_col, legend_col = st.columns([1.25, 1], gap="large")
            with map_col:
                fig = px.choropleth(
                    map_df,
                    geojson=india_geojson,
                    locations="map_name",
                    featureidkey="properties.name",
                    color="cluster",
                    hover_name="map_name",
                    category_orders={"cluster": cluster_order + ["No data"]},
                    color_discrete_map=state_color_map,
                )
                fig.update_geos(
                    fitbounds="locations",
                    showcountries=False,
                    showcoastlines=True,
                    coastlinecolor=chart["boundary"],
                    showland=True,
                    landcolor=chart["land"],
                    showocean=True,
                    oceancolor=chart["ocean"],
                    bgcolor="rgba(0,0,0,0)",
                    showframe=False,
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor=chart["surface"],
                    plot_bgcolor=chart["surface"],
                    height=state_map_height,
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch", theme=None)

            with legend_col:
                rows = "".join(
                    f'<div class="mob-legend-row">'
                    f'<span class="mob-dot" style="background:{pal["cluster"][c]}"></span>'
                    f'<div><div class="mob-legend-name">{desc[c][0]}</div>'
                    f'<div class="mob-legend-meta">'
                    f'{int(np.bincount(labels, minlength=k)[c])} regions</div></div></div>'
                    for c in range(k)
                )
                st.markdown(
                    f'<div class="mob-card">'
                    f'<div class="mob-card-title">Map legend</div>'
                    f'<div class="mob-legend">{rows}'
                    f'<div class="mob-legend-row">'
                    f'<span class="mob-dot" style="background:{chart["no_data"]}"></span>'
                    f'<div><div class="mob-legend-name">No data</div>'
                    f'<div class="mob-legend-meta">No matching regional profile</div>'
                    f'</div></div></div>'
                    f'<p class="mob-card-text" style="margin-top:0.8rem">'
                    f'Grey territories have no matching regional cluster data. The '
                    f'historical Daman and Diu region is not assigned to the newer '
                    f'merged Dadra and Nagar Haveli and Daman and Diu boundary.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            unmatched = sorted(set(state_clusters) - set(boundary_names))
            if unmatched:
                st.caption("Not mapped: " + ", ".join(unmatched))

    elif scope != "national":
        st.markdown(
            '<div class="mob-note">The map is available at the country level. '
            'Boundary files for Indian subnational boundaries are not included '
            'yet.</div>',
            unsafe_allow_html=True,
        )

    else:
        iso = country_iso3()

        if iso is None:
            st.warning("Install pycountry to enable the map: pip install pycountry")
        else:
            map_df = pd.DataFrame({"country": base.index,
                                   "cluster": [desc[c][0] for c in labels]})
            map_df["iso3"] = map_df["country"].map(iso)
            shown = map_df.dropna(subset=["iso3"])

            with st.container(key="card-geo-controls"):
                controls = st.columns([1.9, 1, 1], gap="large")
                controls[0].markdown(
                    '<div class="mob-card-title">Country clusters</div>'
                    '<p class="mob-card-text">Choropleth of every country with a '
                    'complete mobility profile.</p>',
                    unsafe_allow_html=True,
                )
                map_view = controls[1].selectbox(
                    "Map view", ["Flat Map", "Rotatable Globe"],
                    key="mobility_map_view",
                )
                map_height = controls[2].slider(
                    "Map height", 360, 640, 460, 40, key="mobility_map_height"
                )

            projection_type = (
                "natural earth" if map_view == "Flat Map" else "orthographic"
            )

            map_col, legend_col = st.columns([1.75, 1], gap="large")
            with map_col:
                fig = px.choropleth(
                    shown,
                    locations="iso3",
                    color="cluster",
                    hover_name="country",
                    category_orders={"cluster": cluster_order},
                    color_discrete_sequence=pal["cluster"],
                    projection=projection_type,
                )
                fig.update_geos(
                    showcountries=True,
                    countrycolor=chart["boundary"],
                    showland=True,
                    landcolor=chart["land"],
                    showocean=True,
                    oceancolor=chart["ocean"],
                    bgcolor="rgba(0,0,0,0)",
                    projection_rotation_lon=0,
                    showframe=False,
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor=chart["surface"],
                    plot_bgcolor=chart["surface"],
                    height=map_height,
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch", theme=None)

            with legend_col:
                rows = "".join(
                    f'<div class="mob-legend-row">'
                    f'<span class="mob-dot" style="background:{pal["cluster"][c]}"></span>'
                    f'<div><div class="mob-legend-name">{desc[c][0]}</div>'
                    f'<div class="mob-legend-meta">'
                    f'{int(np.bincount(labels, minlength=k)[c])} regions</div></div></div>'
                    for c in range(k)
                )
                st.markdown(
                    f'<div class="mob-card">'
                    f'<div class="mob-card-title">Map legend</div>'
                    f'<div class="mob-legend">{rows}</div>'
                    f'<p class="mob-card-text" style="margin-top:0.8rem">'
                    f'Grey countries have no cluster: Google did not publish data for '
                    f'them, or they were dropped for too much missing data or an '
                    f'incomplete profile.</p>'
                    f'<p class="mob-card-text">'
                    f'Clusters are numbered from the mildest first-wave dip (0) to the '
                    f'deepest. The map shows where behaviour was similar, not why.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            unmatched = map_df[map_df["iso3"].isna()]["country"].tolist()
            if unmatched:
                st.caption("Not drawn (no ISO code found): " + ", ".join(unmatched))
# ----------------------------------------------------------------- validation
with tab_evidence:
    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">01</span>'
        'Model-selection metrics</div>',
        unsafe_allow_html=True,
    )
    st.caption("Precomputed grid from src/cluster.py. Read each algorithm's own curve: "
               "Euclidean and Manhattan values are in different units, so compare where "
               "each line bends or peaks, not the gap between the two lines.")

    with st.expander("How to read this tab", expanded=False):
        st.markdown("""
There are no labels in this problem, so quality is judged from the data itself:
- **Silhouette** (higher is better): how well each region fits its own cluster compared with the nearest other one. About 0.2 to 0.4 means overlapping groups, not sharply separated ones.
- **Davies-Bouldin** (lower is better): how compact and separated the clusters are.
- **WCSS** (look for the elbow): total spread inside clusters. It always falls as k grows, so look for where the drop slows down.
- **Stability** (closer to 1 is better): repeat the clustering on random 80% subsamples and measure how often the same groups reappear.
- A good k is one that looks reasonable on several of these at once, not the maximum of a single one.
""")

    metrics = read_table("clustering_metrics.csv")
    cand = read_table(f"candidates_{scope}.csv")
    m = None
    c = None

    if metrics is not None:
        m = metrics[(metrics["scope"] == scope) & (metrics["feature_set"] == feature_set)
                    & (metrics["scaler"] == scaler)]
        chart_specs = [
            ("silhouette", "Silhouette", "Higher is better"),
            ("davies_bouldin", "Davies–Bouldin", "Lower is better"),
            ("wcss", "WCSS", "Look for the elbow"),
        ]
        metric_cols = st.columns(3, gap="medium")
        for col, (field, title, hint) in zip(metric_cols, chart_specs):
            f = px.line(m, x="k", y=field, color="distance", title=title,
                        markers=True, line_dash="distance",
                        color_discrete_sequence=pal["series"])
            f.add_vline(x=k, line_dash="dot", line_color=chart["muted"])
            f.update_layout(
                height=300,
                paper_bgcolor=chart["surface"],
                plot_bgcolor=chart["surface"],
                margin=dict(l=8, r=8, t=42, b=56),
                title=dict(font=dict(size=12.5)),
                legend=dict(orientation="h", yanchor="top", y=-0.18,
                            xanchor="left", x=0, title_text=""),
            )
            with col:
                st.plotly_chart(f, width="stretch", theme=None)

        st.markdown(
            '<p class="mob-chart-note">Silhouette is better when higher; '
            'Davies–Bouldin is better when lower; WCSS always falls as k grows, '
            'so look for where the decline slows. The dotted line marks the k '
            'selected in the sidebar.</p>',
            unsafe_allow_html=True,
        )

    if cand is not None:
        st.markdown(
            '<div class="mob-section"><span class="mob-section-num">02</span>'
            'Stability</div>',
            unsafe_allow_html=True,
        )
        c = cand[(cand["feature_set"] == feature_set) & (cand["scaler"] == scaler)]
        f = px.line(c, x="k", y="stability_ari", color="distance",
                    title="Mean ARI over 80% subsamples (closer to 1 is more stable)",
                    markers=True, line_dash="distance",
                    color_discrete_sequence=pal["series"])
        f.add_vline(x=k, line_dash="dot", line_color=chart["muted"])
        f.update_layout(
            height=330,
            paper_bgcolor=chart["surface"],
            plot_bgcolor=chart["surface"],
            margin=dict(l=10, r=10, t=44, b=58),
            title=dict(font=dict(size=12.5)),
            legend=dict(orientation="h", yanchor="top", y=-0.18,
                        xanchor="left", x=0, title_text=""),
        )
        st.plotly_chart(f, width="stretch", theme=None)

    if m is not None and c is not None:
        sel = m[m["k"] == k][["distance", "silhouette", "davies_bouldin", "min_cluster_size"]]
        sel = sel.merge(c[c["k"] == k][["distance", "stability_ari"]], on="distance", how="left")
        sel = sel.round(3).rename(columns={
            "distance": "Distance",
            "silhouette": "Silhouette",
            "davies_bouldin": "Davies–Bouldin",
            "min_cluster_size": "Smallest cluster",
            "stability_ari": "Stability (ARI)",
        })
        sel["Distance"] = sel["Distance"].str.title()

        st.markdown(
            f'<div class="mob-section"><span class="mob-section-num">03</span>'
            f'Metrics at k = {k}</div>',
            unsafe_allow_html=True,
        )
        table_col, guide_col = st.columns([1.45, 1], gap="large")
        with table_col:
            st.dataframe(sel, hide_index=True, width="stretch")
        with guide_col:
            st.markdown(
                '<div class="mob-card">'
                '<div class="mob-card-title">How to judge these numbers</div>'
                '<div class="mob-def">'
                '<div><div class="mob-def-term">Silhouette</div>'
                '<div class="mob-def-text">0.2 to 0.4 means clusters overlap '
                'rather than forming sharply separated categories.</div></div>'
                '<div><div class="mob-def-term">Davies–Bouldin</div>'
                '<div class="mob-def-text">Lower values mean clusters are more '
                'compact and further apart.</div></div>'
                '<div><div class="mob-def-term">Stability (ARI)</div>'
                '<div class="mob-def-text">Agreement across subsamples. Values '
                'closer to 1 mean the same groups reappear.</div></div>'
                '<div><div class="mob-def-term">Smallest cluster</div>'
                '<div class="mob-def-text">A very small cluster can indicate that '
                'k is too high for the data.</div></div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

    hold = read_table(f"holdout_{scope}.csv")
    if hold is not None:
        st.markdown(
            '<div class="mob-section"><span class="mob-section-num">04</span>'
            'Out-of-sample check</div>',
            unsafe_allow_html=True,
        )
        st.caption("Prediction strength of about 0.8 or more means the groups found on training "
                   "regions are also present among regions the model never saw. Mean and "
                   "10th percentile over 100 random splits (K-Means, parks excluded, robust "
                   "scaling). Fit on 70% of regions, tested on the unseen 30%.")
        st.dataframe(
            hold.rename(columns={
                "ps_mean": "Prediction strength (mean)",
                "ps_p10": "Prediction strength (10th pct)",
                "ari_mean": "ARI (mean)",
                "ari_p10": "ARI (10th pct)",
            }),
            hide_index=True, width="stretch",
        )
# ------------------------------------------------------------------ over time
with tab_time:
    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">01</span>'
        'Behavioural states over pandemic periods</div>',
        unsafe_allow_html=True,
    )
    st.caption("Each region-period is assigned a state, ordered from most suppressed "
               "activity (0) upward. 'Above baseline' is relative to a Jan–Feb 2020 "
               "reference, not proof of higher mobility than before the pandemic.")

    names = state_labels(scope)
    share_file = first_match(f"state_share_by_period_{scope}_k*.csv")
    trans_file = first_match(f"transition_matrix_{scope}_k*.csv")

    if share_file is not None:
        share_raw = pd.read_csv(share_file, index_col=0)
        share = share_raw.copy()
        share.columns = [names.get(int(c), f"State {c}") for c in share.columns]
        color_map = {n: pal["state"][n.split(": ", 1)[1]] for n in names.values()}

        share.index = [str(p).replace("_", " ").capitalize() for p in share.index]
        fig = px.bar(share, barmode="stack", color_discrete_map=color_map,
                     labels={"value": "Share of regions", "index": "Period"})
        fig.update_layout(
            height=450,
            paper_bgcolor=chart["surface"],
            plot_bgcolor=chart["surface"],
            margin=dict(l=10, r=10, t=14, b=112),
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", yanchor="top", y=-0.26,
                        xanchor="left", x=0, title_text=""),
        )
        fig.update_xaxes(tickangle=-20, tickfont=dict(size=10))

        suppressed = share_raw.iloc[:, 0]
        above_cols = [col for col in share_raw.columns
                      if "above" in names.get(int(col), "").lower()]
        end_share = share_raw.iloc[-1][above_cols].sum()
        peak_period = str(suppressed.idxmax()).replace("_", " ").capitalize()
        last_period = str(share_raw.index[-1]).replace("_", " ").capitalize()

        bar_col, summary_col = st.columns([1.72, 1], gap="large")
        with bar_col:
            st.plotly_chart(fig, width="stretch", theme=None)
        with summary_col:
            st.markdown(
                f'<div class="mob-card">'
                f'<div class="mob-card-title">What stands out</div>'
                f'<div class="mob-def">'
                f'<div><div class="mob-def-term">Most suppressed period</div>'
                f'<div class="mob-def-text">{peak_period} · '
                f'{suppressed.max():.0%} of regions in the suppressed state.</div></div>'
                f'<div><div class="mob-def-term">By {last_period}</div>'
                f'<div class="mob-def-text">{end_share:.0%} of regions are above their '
                f'baseline level.</div></div>'
                f'</div>'
                f'<p class="mob-card-text" style="margin-top:0.8rem">'
                f'Share of regions in each state, stacked by period.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if trans_file is not None:
        tm = pd.read_csv(trans_file, index_col=0)
        tm.index = [names.get(int(i), f"State {i}") for i in tm.index]
        tm.columns = [names.get(int(col), f"State {col}") for col in tm.columns]

        st.markdown(
            '<div class="mob-section"><span class="mob-section-num">02</span>'
            'State transitions</div>',
            unsafe_allow_html=True,
        )
        matrix_col, guide_col = st.columns([1.18, 1], gap="large")
        with matrix_col:
            fig = px.imshow(
                tm, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1,
                labels=dict(x="State in next period", y="State in this period"),
            )
            fig.update_layout(
                height=460,
                paper_bgcolor=chart["surface"],
                plot_bgcolor=chart["surface"],
                margin=dict(l=10, r=10, t=14, b=110),
                coloraxis_colorbar=dict(title="Probability", thickness=11, len=0.8),
            )
            fig.update_xaxes(tickangle=-32, tickfont=dict(size=9.5),
                             title_standoff=6)
            st.plotly_chart(fig, width="stretch", theme=None)
        with guide_col:
            st.markdown(
                '<div class="mob-card">'
                '<div class="mob-card-title">Reading the matrix</div>'
                '<p class="mob-card-text">'
                'Each row shows where regions in that state went in the next period. '
                'A high value on the diagonal means regions tend to stay in the same '
                'state; off-diagonal values show movement between states.'
                '</p>'
                '<p class="mob-card-text">'
                'Rows sum to one. States are ordered from the most suppressed activity '
                'to the most active, so movement toward the lower-right indicates '
                'recovery.'
                '</p>'
                '</div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------- compare regions
with tab_compare:
    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">01</span>'
        'Daily trajectories</div>',
        unsafe_allow_html=True,
    )

    daily = get_daily(scope)
    regions = sorted(daily["region_id"].unique())
    clustered = [r for r in regions if r in base.index]

    with st.container(key="card-compare-controls"):
        controls = st.columns([2.4, 1.2], gap="large")
        chosen = controls[0].multiselect("Regions", regions, default=clustered[:2])
        category = controls[1].selectbox(
            "Category", CATEGORIES, index=CATEGORIES.index("workplaces")
        )

    if chosen:
        wide = (daily[daily["region_id"].isin(chosen)]
                .pivot(index="date", columns="region_id", values=category)
                .rolling(7, min_periods=3).mean())
        fig = px.line(wide, color_discrete_sequence=pal["series"],
                      labels={"value": "% change from baseline (7-day mean)", "date": ""})
        for i, (name, (start, end)) in enumerate(PERIODS.items()):
            if i % 2 == 0:
                fig.add_vrect(x0=start, x1=end, fillcolor=chart["muted"], opacity=0.07,
                              line_width=0,
                              annotation_text=name.replace("_", " ").capitalize(),
                              annotation_position="top left", annotation_font_size=9.5,
                              annotation_font_color=chart["muted"])
        fig.update_layout(
            height=440,
            paper_bgcolor=chart["surface"],
            plot_bgcolor=chart["surface"],
            margin=dict(l=10, r=10, t=32, b=64),
            legend=dict(orientation="h", yanchor="top", y=-0.18,
                        xanchor="left", x=0, title_text=""),
        )
        st.plotly_chart(fig, width="stretch", theme=None)
        st.markdown(
            '<p class="mob-chart-note">Shaded bands: pre-pandemic, reopening 2020, '
            'delta wave and late pandemic. The unshaded gaps between them are the '
            'first wave, winter wave and omicron wave.</p>',
            unsafe_allow_html=True,
        )

        lab = pd.Series(labels, index=base.index)
        lines = [f"**{r}**: {desc[int(lab[r])][0]}" for r in chosen if r in base.index]
        outside = [r for r in chosen if r not in base.index]

        st.markdown(
            '<div class="mob-section"><span class="mob-section-num">02</span>'
            'Profile values</div>',
            unsafe_allow_html=True,
        )
        table_col, side_col = st.columns([1.55, 1], gap="large")

        idx = base.index.intersection(chosen)
        with table_col:
            if len(idx):
                comparison = base.loc[idx].round(2).T
                readable_features = []
                for feature in comparison.index:
                    label = feature.replace("__", " - ").replace("_", " ").title()
                    label = label.replace("Retail And Recreation", "Retail & Recreation")
                    label = label.replace("Grocery And Pharmacy", "Grocery & Pharmacy")
                    label = label.replace("Transit Stations", "Transit")
                    label = label.replace("First Wave Mean", "First wave")
                    label = label.replace("Late Mean", "Late period")
                    readable_features.append(label)

                comparison.index = readable_features
                comparison.loc["Cluster"] = lab.loc[idx].astype(int)

                comparison_styler = comparison.style.format("{:.2f}")
                comparison_styler = comparison_styler.format(
                    "{:.0f}", subset=(["Cluster"], slice(None))
                )
                st.dataframe(comparison_styler, width="stretch")

        with side_col:
            body = "".join(
                f'<div class="mob-def"><div class="mob-def-term">{line.split("**: ", 1)[0].replace("**", "")}</div>'
                f'<div class="mob-def-text">{line.split("**: ", 1)[1]}</div></div>'
                for line in lines
            )
            extra = ""
            if outside:
                extra = (f'<p class="mob-card-text" style="margin-top:0.8rem">'
                         f'Not clustered (incomplete profile or excluded outlier), so no '
                         f'profile is shown for: {", ".join(outside)}.</p>')
            st.markdown(
                f'<div class="mob-card">'
                f'<div class="mob-card-title">Selected regions</div>'
                f'{body}{extra}</div>',
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------- method & limits
with tab_method:
    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">01</span>'
        'How this was built</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mob-grid-2">
            <div class="mob-card">
                <div class="mob-card-title">Data preparation</div>
                <p class="mob-card-text">
                    Daily Google Community Mobility Reports provide percentage changes
                    from a weekday-specific January–February 2020 baseline across six
                    place categories. The data were organised into daily regional series
                    and preprocessed before feature engineering.
                </p>
                <p class="mob-card-text">
                    Internal gaps of up to 14 days were interpolated, with edge filling
                    limited to 7 days. A region was excluded if any feature exceeded the
                    configured 30% missing-data threshold.
                </p>
                <p class="mob-card-text">
                    Each regional profile contains first-wave and late-period means for
                    the six mobility categories, along with trough depth and recovery
                    duration.
                </p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Profiles, clustering and validation</div>
                <p class="mob-card-text">
                    Recovery is measured using a composite of retail, transit and
                    workplace mobility, smoothed with a 7-day rolling mean. Recovery is
                    the first post-trough return to within 10 percentage points below
                    baseline. Regions that do not reach this threshold during the
                    observation window are marked as right-censored.
                </p>
                <p class="mob-card-text">
                    The profiles were standardised and explored using K-Means and
                    K-Medians. Evaluation included WCSS, Silhouette, Davies–Bouldin,
                    subsample stability, hold-out validation and comparison with
                    hierarchical clustering.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">02</span>Glossary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mob-card"><div class="mob-def">'
        '<div><div class="mob-def-term">Baseline</div>'
        '<div class="mob-def-text">Google\'s reference level, calculated as the median '
        'for the same weekday between 3 January and 6 February 2020. Values are '
        'percentage changes relative to this reference.</div></div>'
        '<div><div class="mob-def-term">Cluster</div>'
        '<div class="mob-def-text">A group of regions with similar engineered mobility '
        'profiles. These are exploratory groupings, not official regional '
        'categories.</div></div>'
        '<div><div class="mob-def-term">Behavioural state</div>'
        '<div class="mob-def-text">A label assigned to a region-period observation '
        'through a separate temporal clustering analysis.</div></div>'
        '<div><div class="mob-def-term">Stability (ARI)</div>'
        '<div class="mob-def-text">Agreement between cluster assignments across '
        'repeated subsamples, measured using the Adjusted Rand Index.</div></div>'
        '<div><div class="mob-def-term">Right-censored recovery</div>'
        '<div class="mob-def-text">The region did not reach the recovery threshold '
        'before the available observation period ended.</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mob-section"><span class="mob-section-num">03</span>'
        'Read the results responsibly</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mob-grid-2">
            <div class="mob-card">
                <div class="mob-card-title">Exploratory clusters</div>
                <p class="mob-card-text">Silhouette scores around 0.2 to 0.4 indicate
                overlapping groups rather than sharply separated categories.</p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Baseline limitations</div>
                <p class="mob-card-text">Values represent changes against a fixed
                pre-pandemic reference, not absolute mobility. Seasonality and baseline
                effects, particularly for parks, can influence comparisons.</p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Coverage bias</div>
                <p class="mob-card-text">Google mobility data reflect aggregated data
                from users who enabled Location History. They should not be treated as a
                complete or representative measure of the movement of every person in a
                region.</p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Missing-data exclusions</div>
                <p class="mob-card-text">Regions were removed according to the
                configured missing-data threshold. This can affect which regions remain
                available for analysis.</p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Outlier exclusions</div>
                <p class="mob-card-text">Four regions were excluded from model fitting
                using a robust z-score threshold greater than 10.</p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Hold-out validation</div>
                <p class="mob-card-text">Mean prediction strength did not reach the 0.8
                benchmark in any of the four tested scopes. Cluster reproducibility on
                unseen regions is therefore limited and should not be overstated.</p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Fixed period windows</div>
                <p class="mob-card-text">Common calendar windows may not align with the
                timing of pandemic waves in every country or region.</p>
            </div>
            <div class="mob-card">
                <div class="mob-card-title">Descriptive, not causal</div>
                <p class="mob-card-text">The analysis identifies patterns and
                associations. It does not establish why mobility changed.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mob-meta-line" style="margin-top:1.4rem">'
        '<span><b>Data</b></span>'
        '<span>Google LLC, COVID-19 Community Mobility Reports</span>'
        '<span>February 2020 to October 2022</span>'
        '</div>',
        unsafe_allow_html=True,
    )
