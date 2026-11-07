"""Evaluation: bootstrap stability, cross-configuration agreement, PCA, profiles."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from src.cluster import (FEATURE_SETS, SCALERS, fit_euclidean, kmedians,
                         load_config, load_features)

DISTANCES = ["euclidean", "manhattan"]


def fit_labels(X, k, distance, n_init, seed):
    if distance == "euclidean":
        return fit_euclidean(X, k, n_init, seed)[0]
    return kmedians(X, k, n_init, seed)[0]


def stability(X, k, distance, ref_labels, n_boot, frac, seed):
    """Mean ARI between full-data labels and labels from repeated 80% subsamples."""
    rng = np.random.default_rng(seed)
    n = len(X)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, int(frac * n), replace=False)
        labels = fit_labels(X[idx], k, distance, 3, int(rng.integers(1_000_000_000)))
        scores.append(adjusted_rand_score(ref_labels[idx], labels))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="national")
    ap.add_argument("--feature_set", default="no_parks", choices=FEATURE_SETS)
    ap.add_argument("--scaler", default="robust", choices=list(SCALERS))
    ap.add_argument("--distance", default="euclidean", choices=DISTANCES)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    figs = Path(cfg["paths"]["figures"]) / "evaluation"
    figs.mkdir(parents=True, exist_ok=True)

    seed = cfg["project"]["random_seed"]
    n_init = cfg["clustering"]["n_init"]
    n_boot = args.n_boot or cfg["clustering"]["bootstrap_runs"]
    k_values = list(range(2, 9))
    if args.k not in k_values:
        raise ValueError("--k must be between 2 and 8")

    bases, scaled, labels_store, rows = {}, {}, {}, []
    for fset in FEATURE_SETS:
        base = load_features(processed / f"{args.scope}_features_region.parquet", fset)
        bases[fset] = base
        for sname, Scaler in SCALERS.items():
            X = Scaler().fit_transform(base.to_numpy())
            scaled[(fset, sname)] = X
            for dist in DISTANCES:
                for k in k_values:
                    ref = fit_labels(X, k, dist, n_init, seed)
                    labels_store[(fset, sname, dist, k)] = ref
                    m, s = stability(X, k, dist, ref, n_boot, 0.8, seed)
                    rows.append(dict(feature_set=fset, scaler=sname, distance=dist,
                                     k=k, stability_ari=m, stability_std=s))
                print(f"stability done: {fset} | {sname}")

    stab = pd.DataFrame(rows)
    metrics = pd.read_csv(tables / "clustering_metrics.csv")
    metrics = metrics[metrics["scope"] == args.scope]
    merged = stab.merge(
        metrics[["feature_set", "scaler", "distance", "k", "wcss", "silhouette",
                 "davies_bouldin", "min_cluster_size"]],
        on=["feature_set", "scaler", "distance", "k"])
    merged.round(3).to_csv(tables / f"candidates_{args.scope}.csv", index=False)

    sel = merged[(merged["feature_set"] == args.feature_set) & (merged["scaler"] == args.scaler)]
    print(f"\n=== {args.scope} | {args.feature_set} | {args.scaler} ===\n")
    print(sel.pivot(index="k", columns="distance",
                    values=["silhouette", "davies_bouldin", "stability_ari"]).round(3).to_string())
    print("\nTOP 10 BY STABILITY (k between 3 and 6)\n")
    top = merged[merged["k"].between(3, 6)].sort_values("stability_ari", ascending=False).head(10)
    print(top[["feature_set", "scaler", "distance", "k", "stability_ari",
               "silhouette", "davies_bouldin", "min_cluster_size"]].round(3).to_string(index=False))

    # Agreement between configurations at the chosen k
    cfgs = [(f, s, d) for f in FEATURE_SETS for s in SCALERS for d in DISTANCES]
    names = [f"{f}|{s}|{d[:3]}" for f, s, d in cfgs]
    M = np.array([[adjusted_rand_score(labels_store[(*a, args.k)], labels_store[(*b, args.k)])
                   for b in cfgs] for a in cfgs])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pd.DataFrame(M, index=names, columns=names), annot=True, fmt=".2f",
                cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Agreement between configurations (ARI), {args.scope}, k={args.k}")
    fig.tight_layout()
    fig.savefig(figs / f"ari_agreement_{args.scope}_k{args.k}.png", dpi=130)
    plt.close(fig)

    # PCA and profiles for the chosen configuration
    base = bases[args.feature_set]
    X = scaled[(args.feature_set, args.scaler)]
    labels = labels_store[(args.feature_set, args.scaler, args.distance, args.k)]
    order = (base.assign(c=labels).groupby("c")["trough_depth"].mean()
             .sort_values(ascending=False).index)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[v] for v in labels])

    pca = PCA(n_components=min(X.shape)).fit(X)
    evr = pca.explained_variance_ratio_
    Z = pca.transform(X)[:, :2]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(evr) + 1), evr, label="per component")
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr), color="black", marker="o", label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"PCA scree, {args.scope} | {args.feature_set} | {args.scaler}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / f"pca_scree_{args.scope}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=55, label=f"Cluster {c} (n={m.sum()})")
    for i in np.argsort(-np.abs(Z).sum(axis=1))[:8]:
        ax.annotate(base.index[i], (Z[i, 0], Z[i, 1]), fontsize=8)
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    ax.set_title(f"PCA view: {args.scope} | {args.feature_set} | {args.scaler} | "
                 f"{args.distance} | k={args.k}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / f"pca_scatter_{args.scope}_k{args.k}.png", dpi=130)
    plt.close(fig)

    loadings = pd.DataFrame(pca.components_[:3].T, index=base.columns,
                            columns=["PC1", "PC2", "PC3"]).round(3)
    loadings.to_csv(tables / f"pca_loadings_{args.scope}.csv")
    print("\nPCA explained variance (first 4):", np.round(evr[:4], 3))
    print("\nPCA loadings:\n", loadings.to_string())

    profile = base.assign(cluster=labels).groupby("cluster").mean().round(1)
    profile.insert(0, "n_regions", pd.Series(labels).value_counts().sort_index().values)
    profile.to_csv(tables / f"cluster_profiles_{args.scope}_k{args.k}.csv")
    print("\nCLUSTER PROFILES (raw feature means):\n")
    print(profile.T.to_string())

    members = pd.DataFrame({"region_id": base.index, "cluster": labels}).sort_values(["cluster", "region_id"])
    members.to_csv(tables / f"cluster_members_{args.scope}_k{args.k}.csv", index=False)
    for c, g in members.groupby("cluster"):
        print(f"\nCluster {c} ({len(g)}): " + ", ".join(g["region_id"].head(15)))


if __name__ == "__main__":
    main()