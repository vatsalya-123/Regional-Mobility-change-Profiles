"""Dashboard:  python -m streamlit run app/streamlit_app.py  (run from project root)"""
import os
import sys
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
from app.style import (
    CLUSTER_COLORS,
    SERIES_COLORS,
    STATE_COLORS,
    apply_dashboard_style,
)

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
LINE_KW = dict(markers=True, line_dash="distance",
               color_discrete_sequence=SERIES_COLORS)

st.set_page_config(page_title="Mobility change profiles", layout="wide")
apply_dashboard_style()

@st.cache_data(show_spinner=False)
def get_features(scope, feature_set):
    return load_features(PROCESSED / f"{scope}_features_region.parquet", feature_set)


@st.cache_data(show_spinner="Clustering...")
def run_clustering(scope, feature_set, scaler, distance, k):
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
st.sidebar.title("Settings")
scope_label = st.sidebar.selectbox("Region level", list(SCOPES))
scope = SCOPES[scope_label]
feature_set = st.sidebar.radio("Feature set", ["no_parks", "full"],
                               help="Parks is seasonal (baseline is Jan-Feb 2020), so it is excluded by default.")
scaler = st.sidebar.radio("Scaler", ["robust", "standard"])
distance = st.sidebar.radio("Algorithm", ["euclidean", "manhattan"],
                            format_func=lambda d: "K-Means (Euclidean)" if d == "euclidean"
                            else "K-Medians (Manhattan)")
k = st.sidebar.slider("Number of clusters (k)", 2, 8, 3)

base = get_features(scope, feature_set)
labels, Z, evr = run_clustering(scope, feature_set, scaler, distance, k)
desc = describe_clusters(base, labels)
cluster_order = [desc[i][0] for i in range(k)]

st.title("Regional Mobility Change Profiles")
st.caption(
    "Exploratory clustering of regional mobility patterns during COVID-19 | "
    "Google Community Mobility Reports | Feb 2020 to Oct 2022"
)

tab_overview, tab1, tab_map, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Clusters", "Map", "Validation",
     "Over time", "Compare regions", "About and limits"]
)

# ------------------------------------------------------------------ overview
with tab_overview:
    st.subheader("Research overview")

    overview_cols = st.columns(3, gap="medium")

    with overview_cols[0]:
        with st.container(height=220, border=True):
            st.markdown("**01 | Research question**")
            st.write(
                "Can regions be grouped into distinct mobility-change profiles "
                "based on how public activity changed during the COVID-19 pandemic?"
            )

    with overview_cols[1]:
        with st.container(height=220, border=True):
            st.markdown("**02 | Dataset**")
            st.write(
                "Google COVID-19 Community Mobility Reports, covering February "
                "2020 to October 2022. Analysis is available at national, "
                "Indian state, Indian district and global state levels."
            )

    with overview_cols[2]:
        with st.container(height=220, border=True):
            st.markdown("**03 | Analytical approach**")
            st.write(
                "Data preprocessing, feature scaling, K-Means and K-Medians "
                "clustering, PCA visualisation and cluster validation."
            )

    st.markdown("### Current analysis")

    kpi_cols = st.columns(4, gap="medium")

    with kpi_cols[0]:
        st.metric("Regions analysed", f"{len(base):,}")

    with kpi_cols[1]:
        st.metric("Clusters identified", k)

    with kpi_cols[2]:
        st.metric("PCA variance captured", f"{evr.sum():.1%}")

    with kpi_cols[3]:
        method_name = "K-Means" if distance == "euclidean" else "K-Medians"
        st.metric("Clustering method", method_name)

    st.caption(
        f"Configuration: {scope_label} | "
        f"{feature_set.replace('_', ' ').title()} features | "
        f"{scaler.title()} scaling | "
        f"{'Euclidean' if distance == 'euclidean' else 'Manhattan'} distance | "
        f"k = {k}"
    )

    st.markdown("### Research workflow")

    workflow = st.columns(4, gap="small")

    workflow_steps = [
        ("01", "Data preparation", "Cleaning, missing values and feature selection"),
        ("02", "Feature scaling", "Robust and standard scaling"),
        ("03", "Clustering", "K-Means and K-Medians"),
        ("04", "Validation", "Stability, separation and comparison"),
    ]

    for col, (number, title, description) in zip(workflow, workflow_steps):
        with col:
            with st.container(border=True):
                st.caption(f"STEP {number}")
                st.markdown(f"**{title}**")
                st.caption(description)


    st.markdown("### Research insights")

    cluster_counts = np.bincount(labels, minlength=k)
    largest_cluster = int(np.argmax(cluster_counts))
    largest_count = int(cluster_counts[largest_cluster])
    largest_name, largest_description = desc[largest_cluster]
    largest_share = largest_count / len(labels) if len(labels) else 0

    insight_cols = st.columns(3, gap="medium")

    with insight_cols[0]:
        with st.container(border=True):
            st.caption("LARGEST MOBILITY PROFILE")
            st.markdown(f"**{largest_name}**")
            st.metric("Regions in this cluster", largest_count)
            st.caption(largest_description)

    with insight_cols[1]:
        with st.container(border=True):
            st.caption("CLUSTER COVERAGE")
            st.metric("Share of all regions", f"{largest_share:.1%}")
            st.caption(
                f"{largest_count} out of {len(labels)} regions "
                "belong to the largest cluster."
            )

    with insight_cols[2]:
        with st.container(border=True):
            st.caption("PCA VARIANCE BREAKDOWN")
            st.metric("PC1 variance", f"{evr[0]:.1%}")
            st.metric("PC2 variance", f"{evr[1]:.1%}")
            st.caption(f"Combined variance: {evr[:2].sum():.1%}")

    st.caption(
        "Interpretation: The largest cluster represents the most common "
        "mobility-change profile in the selected regional dataset. "
        "PCA variance indicates how much feature variation is represented "
        "by the two-dimensional visualisation."
    )


# ------------------------------------------------------------------ clusters
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Regions clustered", len(base))
    c2.metric("PCA variance shown (PC1 + PC2)", f"{evr.sum():.0%}")
    c3.metric("Smallest cluster", int(np.bincount(labels).min()))

    with st.expander("What each cluster means", expanded=False):
        for c in range(k):
            name, text = desc[c]
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.write(text)

        st.caption("Clusters are numbered from the mildest first-wave dip (0) to the deepest. Names are "
                   "descriptive labels generated from each cluster's averages, not official categories. "
                   "Recovery days for regions that never recovered are counted up to the end of the "
                   "data, so averages are lower bounds. The data shows how behaviour changed, not why.")

    plot_df = pd.DataFrame({"region": base.index, "PC1": Z[:, 0], "PC2": Z[:, 1],
                            "cluster": [desc[c][0] for c in labels]})
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", hover_name="region",
                     color_discrete_sequence=CLUSTER_COLORS,
                     category_orders={"cluster": cluster_order},
                     labels={"PC1": f"PC1 ({evr[0]:.0%})", "PC2": f"PC2 ({evr[1]:.0%})"})
    fig.update_traces(
        marker=dict(size=10, opacity=0.88, line=dict(width=0.6))
    )

    # Show the mean position of each cluster in the PCA projection.
    centres = (
        plot_df.groupby("cluster", sort=False)[["PC1", "PC2"]]
        .mean()
        .reset_index()
    )

    fig.add_scatter(
        x=centres["PC1"],
        y=centres["PC2"],
        mode="markers+text",
        text=centres["cluster"].str.extract(r"Cluster (\d+)")[0].apply(
            lambda value: f"Mean {value}"
        ),
        textposition="top center",
        textfont=dict(color="#FFFFFF", size=11),
        marker=dict(
            symbol="x",
            size=17,
            color="#FFFFFF",
            line=dict(width=2)
        ),
        name="PCA cluster means",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "PC1: %{x:.2f}<br>"
            "PC2: %{y:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="top", y=-0.18,
            xanchor="left", x=0, title_text=""
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,34,53,0.88)",
        margin=dict(l=24, r=24, t=35, b=85),
        height=520,
        hovermode="closest"
    )
    st.plotly_chart(fig, width="stretch")

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
            metric = metric_labels.get(
                metric, metric.replace("_", " ").title()
            )
            readable_labels.append(f"{category} \u2014 {metric}")
        else:
            readable_labels.append(feature.replace("_", " ").title())

    display_profile.index = readable_labels
    display_profile.index.name = "Feature"

    st.subheader("Cluster profile comparison")
    st.caption(
        "Mean feature values within each cluster. "
        "Number of regions indicates the size of each cluster."
    )
    st.dataframe(display_profile, width="stretch")
    with st.expander("How to read the feature names"):
        st.markdown("""
- **first_wave_mean**: average change from the Jan-Feb 2020 baseline during the first wave (Mar-Jun 2020).
- **late_mean**: average change during the late pandemic period (Apr-Oct 2022).
- **trough_depth**: the lowest point of overall activity (retail, transit, workplaces) in the first wave. More negative means a deeper dip.
- **recovery_days**: days from that lowest point until activity was back within 10 points of baseline. Longer means slower recovery.
- **residential**: time spent at home. It moves in the opposite direction to the other categories.
""")

    z = (base.assign(cluster=labels).groupby("cluster").mean() - base.mean()) / base.std()
    z.index = [f"Cluster {c}" for c in z.index]

    heat_data = z.T.round(2)
    heat_data.index = display_profile.index[1:]

    # Wrap long feature labels without changing the underlying values.
    heat_display = heat_data.copy()
    heat_display.index = [
        str(label).replace(" ? ", "<br>")
        for label in heat_display.index
    ]

    heat_height = max(380, min(460, len(heat_display) * 30 + 80))

    heat = px.imshow(
        heat_display,
        color_continuous_scale="RdBu_r",
        zmin=-2.5,
        zmax=2.5,
        text_auto=".2f",
        aspect="auto",
        labels=dict(x="Cluster", y="Feature", color="z-score")
    )

    heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,34,53,0.88)",
        font=dict(color="#E8EEF5"),
        margin=dict(l=20, r=65, t=20, b=45),
        height=heat_height,
        coloraxis_colorbar=dict(
            title="z-score",
            thickness=14,
            len=0.85
        )
    )

    heat.update_xaxes(
        title_text="Cluster",
        side="bottom",
        tickfont=dict(size=12)
    )

    heat.update_yaxes(
        title_text="",
        tickfont=dict(size=11),
        automargin=True
    )

    st.subheader("Cluster signatures (z-scores vs all regions)")
    st.caption(
        "Red indicates above-average feature values; blue indicates "
        "below-average values. Numbers show standardised z-scores."
    )
    st.plotly_chart(heat, width="stretch")

    st.subheader("Cluster membership")
    st.caption(
        "Explore the regions assigned to each mobility profile. "
        "Expand a cluster to view its members."
    )

    members = pd.DataFrame(
        {"region": base.index, "cluster": labels}
    ).sort_values(["cluster", "region"])

    for c in range(k):
        names = members.loc[
            members["cluster"] == c, "region"
        ].astype(str).tolist()

        with st.expander(
            f"{desc[c][0]}  |  {len(names)} regions",
            expanded=False
        ):
            member_cols = st.columns(3, gap="medium")
            split_names = np.array_split(names, 3)

            for col, group in zip(member_cols, split_names):
                with col:
                    st.markdown(
                        "\n".join(f"- {name}" for name in group)
                        if len(group) else "_No regions in this cluster._"
                    )

    st.download_button(
        "Download assignments (CSV)",
        members.to_csv(index=False),
        file_name=f"clusters_{scope}_k{k}.csv"
    )

# ----------------------------------------------------------------------- map
with tab_map:
    st.subheader("Map of country clusters")
    if scope != "national":
        st.info("The map is available at the country level. Boundary files for Indian "
                "subnational boundaries are not included yet.")
    else:
        iso = country_iso3()
        if iso is None:
            st.warning("Install pycountry to enable the map: pip install pycountry")
        else:
            map_df = pd.DataFrame({"country": base.index,
                                   "cluster": [desc[c][0] for c in labels]})
            map_df["iso3"] = map_df["country"].map(iso)
            shown = map_df.dropna(subset=["iso3"])
            map_controls = st.columns([1, 2], gap="medium")

            with map_controls[0]:
                map_view = st.selectbox(
                    "Map view",
                    ["Flat Map", "Rotatable Globe"],
                    key="mobility_map_view"
                )

            with map_controls[1]:
                map_height = st.slider(
                    "Map height",
                    min_value=400,
                    max_value=800,
                    value=560,
                    step=40,
                    key="mobility_map_height"
                )

            projection_type = (
                "natural earth"
                if map_view == "Flat Map"
                else "orthographic"
            )

            fig = px.choropleth(
                shown,
                locations="iso3",
                color="cluster",
                hover_name="country",
                category_orders={"cluster": cluster_order},
                color_discrete_sequence=CLUSTER_COLORS,
                projection=projection_type
            )

            fig.update_geos(
                showcountries=True,
                countrycolor="#52677D",
                showland=True,
                landcolor="#142235",
                showocean=True,
                oceancolor="#0B1220",
                bgcolor="rgba(0,0,0,0)",
                projection_rotation_lon=0
            )

            fig.update_layout(
                margin=dict(l=0, r=0, t=20, b=65),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=map_height,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.08,
                    xanchor="left",
                    x=0,
                    title_text=""
                )
            )

            st.plotly_chart(fig, width="stretch")
            st.caption("Dark grey countries have no cluster: Google did not publish data for them, "
                       "or they were dropped for too much missing data or an incomplete profile. "
                       "Clusters are numbered from the mildest first-wave dip (0) to the deepest. "
                       "The map shows where behaviour was similar, not why.")
            unmatched = map_df[map_df["iso3"].isna()]["country"].tolist()
            if unmatched:
                st.caption("Not drawn (no ISO code found): " + ", ".join(unmatched))

# ---------------------------------------------------------------- validation
with tab2:
    st.subheader("Validation metrics across k")
    with st.expander("How to read this tab", expanded=True):
        st.markdown("""
There are no labels in this problem, so quality is judged from the data itself:
- **Silhouette** (higher is better): how well each region fits its own cluster compared with the nearest other one. About 0.2 to 0.4 means overlapping groups, not sharply separated ones.
- **Davies-Bouldin** (lower is better): how compact and separated the clusters are.
- **WCSS** (look for the elbow): total spread inside clusters. It always falls as k grows, so look for where the drop slows down.
- **Stability** (closer to 1 is better): repeat the clustering on random 80% subsamples and measure how often the same groups reappear.
- A good k is one that looks reasonable on several of these at once, not the maximum of a single one.
""")
    st.caption("Precomputed grid from src/cluster.py. Read each algorithm's own curve: "
               "Euclidean and Manhattan values are in different units, so compare where "
               "each line bends or peaks, not the gap between the two lines.")
    metrics = read_table("clustering_metrics.csv")
    cand = read_table(f"candidates_{scope}.csv")
    m = None
    c = None
    if metrics is not None:
        m = metrics[(metrics["scope"] == scope) & (metrics["feature_set"] == feature_set)
                    & (metrics["scaler"] == scaler)]
        for col, title in [("silhouette", "Silhouette (higher is better)"),
                           ("davies_bouldin", "Davies-Bouldin (lower is better)"),
                           ("wcss", "WCSS / total within-cluster distance (look for the elbow)")]:
            f = px.line(m, x="k", y=col, color="distance", title=title, **LINE_KW)
            f.add_vline(x=k, line_dash="dot")
            f.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,34,53,0.88)",
                margin=dict(l=24, r=24, t=65, b=45),
                height=390
            )
            st.plotly_chart(f, width="stretch")
    if cand is not None:
        c = cand[(cand["feature_set"] == feature_set) & (cand["scaler"] == scaler)]
        f = px.line(c, x="k", y="stability_ari", color="distance",
                    title="Stability: mean ARI over 80% subsamples (closer to 1 is more stable)",
                    **LINE_KW)
        f.add_vline(x=k, line_dash="dot")
        f.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,34,53,0.88)",
            margin=dict(l=24, r=24, t=65, b=45),
            height=390
        )
        st.plotly_chart(f, width="stretch")
    if m is not None and c is not None:
        sel = m[m["k"] == k][["distance", "silhouette", "davies_bouldin", "min_cluster_size"]]
        sel = sel.merge(c[c["k"] == k][["distance", "stability_ari"]], on="distance", how="left")
        st.subheader(f"Metrics at k = {k}")
        st.dataframe(sel.round(3), hide_index=True, width="stretch")
    hold = read_table(f"holdout_{scope}.csv")
    if hold is not None:
        st.subheader("Hold-out validation (fit on 70% of regions, test on the unseen 30%)")
        st.caption("Prediction strength of about 0.8 or more means the groups found on training "
                   "regions are also present among regions the model never saw. Mean and "
                   "10th percentile over 100 random splits (K-Means, parks excluded, robust scaling).")
        st.dataframe(hold.rename(columns={"ps_mean": "prediction strength (mean)",
                                          "ps_p10": "prediction strength (10th pct)",
                                          "ari_mean": "ARI (mean)",
                                          "ari_p10": "ARI (10th pct)"}),
                     hide_index=True, width="stretch")

# ----------------------------------------------------------------- over time
with tab3:
    st.subheader("Behavioural states over pandemic periods")
    st.caption("Each region-period is assigned a state, ordered from most suppressed activity (0) "
               "upward. 'Above baseline' is relative to a Jan-Feb 2020 reference, not proof of "
               "higher mobility than before the pandemic.")
    names = state_labels(scope)
    share_file = first_match(f"state_share_by_period_{scope}_k*.csv")
    trans_file = first_match(f"transition_matrix_{scope}_k*.csv")
    if share_file is not None:
        share_raw = pd.read_csv(share_file, index_col=0)
        share = share_raw.copy()
        share.columns = [names.get(int(c), f"State {c}") for c in share.columns]
        color_map = {n: STATE_COLORS[n.split(": ", 1)[1]] for n in names.values()}
        st.plotly_chart(px.bar(share, barmode="stack", color_discrete_map=color_map,
                               labels={"value": "Share of regions", "index": "Period"}),
                        width="stretch")
        suppressed = share_raw.iloc[:, 0]
        above_cols = [c for c in share_raw.columns if "above" in names.get(int(c), "").lower()]
        st.markdown(
            f"- **Most suppressed period:** {suppressed.idxmax()} "
            f"({suppressed.max():.0%} of regions in the suppressed state).\n"
            f"- **By {share_raw.index[-1]}:** {share_raw.iloc[-1][above_cols].sum():.0%} of "
            f"regions are above their baseline level.")
    if trans_file is not None:
        tm = pd.read_csv(trans_file, index_col=0)
        tm.index = [names.get(int(i), f"State {i}") for i in tm.index]
        tm.columns = [names.get(int(c), f"State {c}") for c in tm.columns]
        st.plotly_chart(px.imshow(tm, text_auto=".2f", color_continuous_scale="Blues",
                                  zmin=0, zmax=1,
                                  labels=dict(x="State in next period", y="State in this period"),
                                  title="Transition probabilities between consecutive periods"),
                        width="stretch")
        st.caption("Each row shows where regions in that state went in the next period. A high "
                   "value on the diagonal means regions tend to stay in the same state.")

# ----------------------------------------------------------- compare regions
with tab4:
    st.subheader("Compare daily trajectories")
    daily = get_daily(scope)
    regions = sorted(daily["region_id"].unique())
    clustered = [r for r in regions if r in base.index]
    chosen = st.multiselect("Regions", regions, default=clustered[:2])
    category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index("workplaces"))
    if chosen:
        wide = (daily[daily["region_id"].isin(chosen)]
                .pivot(index="date", columns="region_id", values=category)
                .rolling(7, min_periods=3).mean())
        fig = px.line(wide, color_discrete_sequence=SERIES_COLORS,
                      labels={"value": "% change from baseline (7-day mean)", "date": ""})
        for i, (name, (start, end)) in enumerate(PERIODS.items()):
            if i % 2 == 0:
                fig.add_vrect(x0=start, x1=end, fillcolor="gray", opacity=0.08, line_width=0,
                              annotation_text=name, annotation_position="top left",
                              annotation_font_size=9)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,34,53,0.88)",
            margin=dict(l=24, r=24, t=45, b=45),
            height=480
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Shaded bands: pre_pandemic, reopening_2020, delta_wave, late_pandemic. "
                   "The unshaded gaps between them are first_wave, winter_wave and omicron_wave.")
        lab = pd.Series(labels, index=base.index)
        lines = [f"**{r}**: {desc[int(lab[r])][0]}" for r in chosen if r in base.index]
        if lines:
            st.markdown("  \n".join(lines))
        outside = [r for r in chosen if r not in base.index]
        if outside:
            st.caption("Not clustered (incomplete profile or excluded outlier), so no profile "
                       "is shown for: " + ", ".join(outside))
        idx = base.index.intersection(chosen)
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

                st.dataframe(comparison, width="stretch")

# ---------------------------------------------------------------- about
with tab5:
    st.markdown("""
### How this was built

Daily Google Community Mobility Reports provide percentage changes from a
weekday-specific January-February 2020 baseline across six place categories.
The data were organised into daily regional series and preprocessed before
feature engineering.

Internal gaps of up to 14 days were interpolated, with edge filling limited
to 7 days. A region was excluded if any feature exceeded the configured
30% missing-data threshold.

Each regional profile contains first-wave and late-period means for the six
mobility categories, along with trough depth and recovery duration.
Recovery is measured using a composite of retail, transit and workplace
mobility, smoothed with a 7-day rolling mean. Recovery is the first post-trough
return to within 10 percentage points below baseline. Regions that do not
reach this threshold during the observation window are marked as
right-censored.

The profiles were standardised and explored using K-Means and K-Medians.
Evaluation included WCSS, Silhouette, Davies-Bouldin, subsample stability,
hold-out validation and comparison with hierarchical clustering.

### Glossary

- **Baseline:** Google's reference level, calculated as the median for the
  same weekday between 3 January and 6 February 2020. Values are percentage
  changes relative to this reference.
- **Cluster:** A group of regions with similar engineered mobility profiles.
  These are exploratory groupings, not official regional categories.
- **Behavioural state:** A label assigned to a region-period observation
  through a separate temporal clustering analysis.
- **Stability (ARI):** Agreement between cluster assignments across repeated
  subsamples, measured using the Adjusted Rand Index.
- **Right-censored recovery:** The region did not reach the recovery threshold
  before the available observation period ended.

### Read the results responsibly

- **Exploratory clusters:** Silhouette scores around 0.2 to 0.4 indicate
  overlapping groups rather than sharply separated categories.
- **Baseline limitations:** Values represent changes against a fixed
  pre-pandemic reference, not absolute mobility. Seasonality and baseline
  effects, particularly for parks, can influence comparisons.
- **Coverage bias:** Google mobility data reflect aggregated data from users
  who enabled Location History. They should not be treated as a complete or
  representative measure of the movement of every person in a region.
- **Missing-data exclusions:** Regions were removed according to the
  configured missing-data threshold. This can affect which regions remain
  available for analysis.
- **Outlier exclusions:** Four regions were excluded from model fitting using
  a robust z-score threshold greater than 10.
- **Hold-out validation:** Mean prediction strength did not reach the 0.8
  benchmark in any of the four tested scopes. Cluster reproducibility on
  unseen regions is therefore limited and should not be overstated.
- **Fixed period windows:** Common calendar windows may not align with the
  timing of pandemic waves in every country or region.
- **Descriptive, not causal:** The analysis identifies patterns and
  associations. It does not establish why mobility changed.

### Data

Google LLC, COVID-19 Community Mobility Reports
(February 2020 to October 2022).
""")
