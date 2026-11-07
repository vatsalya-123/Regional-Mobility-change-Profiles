"""Hierarchical clustering comparison against the chosen K-Means solution."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.cluster import FEATURE_SETS, SCALERS, fit_euclidean, load_config, load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="national")
    ap.add_argument("--feature_set", default="no_parks", choices=FEATURE_SETS)
    ap.add_argument("--scaler", default="robust", choices=list(SCALERS))
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    figs = Path(cfg["paths"]["figures"]) / "evaluation"
    figs.mkdir(parents=True, exist_ok=True)

    base = load_features(processed / f"{args.scope}_features_region.parquet", args.feature_set)
    X = SCALERS[args.scaler]().fit_transform(base.to_numpy())
    ref = fit_euclidean(X, args.k, cfg["clustering"]["n_init"], cfg["project"]["random_seed"])[0]
    order = (base.assign(c=ref).groupby("c")["trough_depth"].mean()
        .sort_values(ascending=False).index)
    ref = np.array([{old: new for new, old in enumerate(order)}[v] for v in ref])

    dists = {"euclidean": pdist(X, "euclidean"), "cityblock": pdist(X, "cityblock")}
    combos = [("ward", "euclidean"), ("average", "euclidean"), ("complete", "euclidean"),
              ("average", "cityblock"), ("complete", "cityblock")]

    rows, trees = [], {}
    for method, metric in combos:
        D = dists[metric]
        Z = linkage(D, method=method)
        trees[(method, metric)] = Z
        coph, _ = cophenet(Z, D)
        labels = fcluster(Z, args.k, criterion="maxclust") - 1
        n_found = len(np.unique(labels))
        sil = (silhouette_score(X, labels, metric="manhattan" if metric == "cityblock" else "euclidean")
               if n_found > 1 else np.nan)
        ct = pd.crosstab(ref, labels).to_numpy()
        r, c = linear_sum_assignment(-ct)
        rows.append(dict(method=method, distance=metric, cophenetic=coph,
                         silhouette=sil, ari_vs_kmeans=adjusted_rand_score(ref, labels),
                         agreement_pct=100 * ct[r, c].sum() / len(ref),
                         n_clusters=n_found,
                         min_cluster_size=int(np.bincount(labels).min())))
        if (method, metric) == ("ward", "euclidean"):
            ward_labels = labels

    res = pd.DataFrame(rows).round(3)
    res.to_csv(tables / f"hierarchical_comparison_{args.scope}_k{args.k}.csv", index=False)
    print(f"\nHIERARCHICAL vs K-MEANS ({args.scope}, k={args.k})\n")
    print(res.to_string(index=False))
    print("\nK-Means (rows) vs Ward (columns):\n")
    print(pd.crosstab(pd.Series(ref, name="kmeans"), pd.Series(ward_labels, name="ward")))

    Z = trees[("ward", "euclidean")]
    fig, ax = plt.subplots(figsize=(10, min(40, max(8, len(X) * 0.17))))
    dendrogram(Z, labels=list(base.index), orientation="right", leaf_font_size=6,
               color_threshold=Z[-(args.k - 1), 2] - 1e-9, ax=ax)
    ax.set_title(f"Ward dendrogram, {args.scope} (cut at k={args.k})")
    fig.tight_layout()
    fig.savefig(figs / f"dendrogram_ward_{args.scope}.png", dpi=130)
    plt.close(fig)
    print("\nSaved dendrogram.")


if __name__ == "__main__":
    main()