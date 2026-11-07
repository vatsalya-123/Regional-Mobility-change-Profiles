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
          "India districts": "india_districts"}

st.set_page_config(page_title="Mobility change profiles", layout="wide")


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

st.title("Regional mobility change profiles during pandemic periods")
st.caption("Google COVID-19 Community Mobility Reports, Feb 2020 to Oct 2022. "
           "Clusters are exploratory profiles, not permanent categories.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Clusters", "Validation", "Over time", "Compare regions", "About and limits"])

# ------------------------------------------------------------------ clusters
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Regions clustered", len(base))
    c2.metric("PCA variance shown (PC1 + PC2)", f"{evr.sum():.0%}")
    c3.metric("Smallest cluster", int(np.bincount(labels).min()))
    st.caption("Clusters are numbered from the mildest first-wave dip (0) to the deepest (highest number).")

    plot_df = pd.DataFrame({"region": base.index, "PC1": Z[:, 0], "PC2": Z[:, 1],
                            "cluster": [f"Cluster {c}" for c in labels]})
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", hover_name="region",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     category_orders={"cluster": [f"Cluster {i}" for i in range(k)]},
                     labels={"PC1": f"PC1 ({evr[0]:.0%})", "PC2": f"PC2 ({evr[1]:.0%})"})
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    st.plotly_chart(fig, width="stretch")

    profile = base.assign(cluster=labels).groupby("cluster").mean().round(1)
    profile.insert(0, "n_regions", np.bincount(labels))
    profile.index = [f"Cluster {c}" for c in profile.index]
    st.subheader("Cluster profiles (raw feature means)")
    st.dataframe(profile.T, width="stretch")

    z = (base.assign(cluster=labels).groupby("cluster").mean() - base.mean()) / base.std()
    z.index = [f"Cluster {c}" for c in z.index]
    heat = px.imshow(z.T.round(2), color_continuous_scale="RdBu_r", zmin=-2.5, zmax=2.5,
                     aspect="auto", labels=dict(x="Cluster", y="", color="z-score"))
    st.subheader("Cluster signatures (z-scores vs all regions)")
    st.plotly_chart(heat, width="stretch")

    members = pd.DataFrame({"region": base.index, "cluster": labels}).sort_values(["cluster", "region"])
    with st.expander("Members of each cluster"):
        for c in range(k):
            names = members.loc[members["cluster"] == c, "region"]
            st.markdown(f"**Cluster {c} ({len(names)})**: " + ", ".join(names))
    st.download_button("Download assignments (CSV)", members.to_csv(index=False),
                       file_name=f"clusters_{scope}_k{k}.csv")

# ---------------------------------------------------------------- validation
with tab2:
    st.subheader("Validation metrics across k")
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
            st.plotly_chart(f, width="stretch")
    if cand is not None:
        c = cand[(cand["feature_set"] == feature_set) & (cand["scaler"] == scaler)]
        f = px.line(c, x="k", y="stability_ari", color="distance",
                    title="Stability: mean ARI over 80% subsamples (closer to 1 is more stable)",
                    **LINE_KW)
        f.add_vline(x=k, line_dash="dot")
        st.plotly_chart(f, width="stretch")
    if m is not None and c is not None:
        sel = m[m["k"] == k][["distance", "silhouette", "davies_bouldin", "min_cluster_size"]]
        sel = sel.merge(c[c["k"] == k][["distance", "stability_ari"]], on="distance", how="left")
        st.subheader(f"Metrics at k = {k}")
        st.dataframe(sel.round(3), hide_index=True, width="stretch")
# ----------------------------------------------------------------- over time
with tab3:
    st.subheader("Behavioural states over pandemic periods")
    st.caption("Each region-period is assigned a state (0 = most suppressed activity). "
               "'Above baseline' is relative to a Jan-Feb 2020 reference, not proof of higher mobility.")
    share_file = first_match(f"state_share_by_period_{scope}_k*.csv")
    trans_file = first_match(f"transition_matrix_{scope}_k*.csv")
    if share_file is not None:
        share = pd.read_csv(share_file, index_col=0)
        share.columns = [f"State {c}" for c in share.columns]
        st.plotly_chart(px.bar(share, barmode="stack",
                               labels={"value": "Share of regions", "index": "Period"}),
                        width="stretch")
    if trans_file is not None:
        tm = pd.read_csv(trans_file, index_col=0)
        st.plotly_chart(px.imshow(tm, text_auto=".2f", color_continuous_scale="Blues",
                                  zmin=0, zmax=1,
                                  labels=dict(x="State in next period", y="State in this period"),
                                  title="Transition probabilities between consecutive periods"),
                        width="stretch")

# ----------------------------------------------------------- compare regions
with tab4:
    st.subheader("Compare daily trajectories")
    daily = get_daily(scope)
    regions = sorted(daily["region_id"].unique())
    default = regions[:2] if len(regions) >= 2 else regions
    chosen = st.multiselect("Regions", regions, default=default)
    category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index("workplaces"))
    if chosen:
        wide = (daily[daily["region_id"].isin(chosen)]
                .pivot(index="date", columns="region_id", values=category)
                .rolling(7, min_periods=3).mean())
        fig = px.line(wide, labels={"value": "% change from baseline (7-day mean)", "date": ""})
        for i, (name, (start, end)) in enumerate(PERIODS.items()):
            if i % 2 == 0:
                fig.add_vrect(x0=start, x1=end, fillcolor="gray", opacity=0.08, line_width=0,
                              annotation_text=name, annotation_position="top left",
                              annotation_font_size=9)
        st.plotly_chart(fig, width="stretch")
        idx = base.index.intersection(chosen)
        if len(idx):
            st.dataframe(base.loc[idx].assign(
                cluster=pd.Series(labels, index=base.index).loc[idx]).T, width="stretch")

# ---------------------------------------------------------------- about
with tab5:
    st.markdown("""
### How this was built
Daily percent-change-from-baseline values for six place categories were cleaned (short gaps
interpolated, regions with too much missing data dropped), turned into region profiles
(first-wave means, late-period means, trough depth, recovery time), standardised, and
clustered with K-Means and K-Medians. Quality was checked with WCSS, Silhouette,
Davies-Bouldin, subsample stability and agreement with hierarchical clustering.

### Read the results responsibly
- **Clusters are exploratory.** Silhouette scores of about 0.2 to 0.4 mean groups overlap on a continuum. They are not permanent categories of regions.
- **Values are changes against a fixed Jan-Feb 2020 baseline**, not absolute mobility. Baseline quirks and seasonality (especially for parks) can inflate later values.
- **Coverage bias:** only users with Location History are included, and small or rural regions are withheld or dropped, so results over-represent larger, better-covered areas.
- **Excluded outliers:** a few regions with implausible values (robust z-score above 10) were removed from model fitting.
- **Fixed period windows** blur country-specific wave timing.
- **Descriptive, not causal.** Nothing here explains why a region behaved as it did.

### Data
Google LLC, COVID-19 Community Mobility Reports (Feb 2020 to Oct 2022).
""")