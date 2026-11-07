"""Clustering experiments: K-Means (Euclidean) and K-Medians (Manhattan)
across k, scalers and feature sets, scored with WCSS, Silhouette and Davies-Bouldin."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

SCOPES = ["national", "india_states", "india_districts", "global_states"]
FEATURE_SETS = ["full", "no_parks"]
SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(path, feature_set):
    df = pd.read_parquet(path).set_index("region_id")
    df = df.drop(columns=["recovery_censored"])
    if feature_set == "no_parks":
        df = df[[c for c in df.columns if not c.startswith("parks__")]]
    return df


def l1_dist(X, centers):
    return np.abs(X[:, None, :] - centers[None, :, :]).sum(axis=2)


def kmedians(X, k, n_init, seed, max_iter=100):
    """K-Medians (Manhattan): assign by L1 distance, update centres by median."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = X[rng.choice(len(X), k, replace=False)]
        for _ in range(max_iter):
            labels = l1_dist(X, centers).argmin(axis=1)
            new = np.array([
                np.median(X[labels == j], axis=0) if (labels == j).any() else centers[j]
                for j in range(k)
            ])
            if np.allclose(new, centers):
                break
            centers = new
        d = l1_dist(X, centers)
        labels = d.argmin(axis=1)
        cost = d[np.arange(len(X)), labels].sum()
        if best is None or cost < best[2]:
            best = (labels, centers, cost)
    return best


def fit_euclidean(X, k, n_init, seed):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
    return km.labels_, km.cluster_centers_, km.inertia_


def davies_bouldin_l1(X, labels, centers):
    k = len(centers)
    s = np.array([
        np.abs(X[labels == j] - centers[j]).sum(axis=1).mean() if (labels == j).any() else 0.0
        for j in range(k)
    ])
    m = np.abs(centers[:, None, :] - centers[None, :, :]).sum(axis=2)
    np.fill_diagonal(m, np.inf)
    return ((s[:, None] + s[None, :]) / m).max(axis=1).mean()


def score(X, labels, centers, distance):
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    if distance == "euclidean":
        return silhouette_score(X, labels), davies_bouldin_score(X, labels)
    return (silhouette_score(X, labels, metric="manhattan"),
            davies_bouldin_l1(X, labels, centers))


def plot_combo(df, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for dist, g in df.groupby("distance"):
        g = g.sort_values("k")
        axes[0].plot(g["k"], g["wcss"] / g["wcss"].iloc[0], marker="o", label=dist)
        axes[1].plot(g["k"], g["silhouette"], marker="o", label=dist)
        axes[2].plot(g["k"], g["davies_bouldin"], marker="o", label=dist)
    axes[0].set_title("WCSS (relative to k=2)")
    axes[1].set_title("Silhouette (higher is better)")
    axes[2].set_title("Davies-Bouldin (lower is better)")
    for ax in axes:
        ax.set_xlabel("k")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    figs = Path(cfg["paths"]["figures"]) / "clustering"
    figs.mkdir(parents=True, exist_ok=True)

    seed = cfg["project"]["random_seed"]
    n_init = cfg["clustering"]["n_init"]
    k_min, k_max = cfg["clustering"]["k_range"]
    rows = []

    for scope in SCOPES:
        for fset in FEATURE_SETS:
            base = load_features(processed / f"{scope}_features_region.parquet", fset)
            for sname, Scaler in SCALERS.items():
                X = Scaler().fit_transform(base.to_numpy())
                combo = []
                for dist in cfg["clustering"]["distances"]:
                    for k in range(k_min, k_max + 1):
                        if dist == "euclidean":
                            labels, centers, cost = fit_euclidean(X, k, n_init, seed)
                        else:
                            labels, centers, cost = kmedians(X, k, n_init, seed)
                        sil, db = score(X, labels, centers, dist)
                        rec = dict(scope=scope, feature_set=fset, scaler=sname,
                                   distance=dist, k=k, n_regions=len(X),
                                   wcss=cost, silhouette=sil, davies_bouldin=db,
                                   min_cluster_size=int(np.bincount(labels, minlength=k).min()))
                        rows.append(rec)
                        combo.append(rec)
                plot_combo(pd.DataFrame(combo), f"{scope} | {fset} | {sname}",
                           figs / f"{scope}__{fset}__{sname}.png")
                print(f"done: {scope} | {fset} | {sname}")

    res = pd.DataFrame(rows)
    res.to_csv(tables / "clustering_metrics.csv", index=False)

    keys = ["scope", "feature_set", "scaler", "distance"]
    best = res.loc[res.groupby(keys)["silhouette"].idxmax(),
                   keys + ["k", "silhouette", "davies_bouldin", "min_cluster_size"]]
    best.round(3).to_csv(tables / "best_k_by_silhouette.csv", index=False)
    print("\nBEST k BY SILHOUETTE (national)\n")
    print(best[best["scope"] == "national"].round(3).to_string(index=False))
    print(f"\nFull grid saved: {tables / 'clustering_metrics.csv'}")


if __name__ == "__main__":
    main()