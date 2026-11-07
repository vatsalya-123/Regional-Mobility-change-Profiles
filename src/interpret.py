"""Interpretation: feature contributions, borderline regions, surrogate rules."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src.cluster import FEATURE_SETS, SCALERS, fit_euclidean, load_config, load_features


def relabel_by_trough(base, labels):
    order = (base.assign(c=labels).groupby("c")["trough_depth"].mean()
             .sort_values(ascending=False).index)
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[v] for v in labels])


def membership_confidence(X, ref, k, n_boot, frac, seed):
    """Share of 80% subsamples in which each region lands in its reference cluster."""
    rng = np.random.default_rng(seed)
    n = len(X)
    agree = np.zeros(n)
    for _ in range(n_boot):
        idx = rng.choice(n, int(frac * n), replace=False)
        km = KMeans(n_clusters=k, n_init=5,
                    random_state=int(rng.integers(1_000_000_000))).fit(X[idx])
        ct = np.zeros((k, k))
        np.add.at(ct, (ref[idx], km.labels_), 1)
        r, c = linear_sum_assignment(-ct)
        mapping = dict(zip(c, r))
        pred = np.array([mapping[p] for p in km.predict(X)])
        agree += (pred == ref)
    return agree / n_boot


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
    seed = cfg["project"]["random_seed"]
    k = args.k

    base = load_features(processed / f"{args.scope}_features_region.parquet", args.feature_set)
    X = SCALERS[args.scaler]().fit_transform(base.to_numpy())
    labels = relabel_by_trough(base, fit_euclidean(X, k, cfg["clustering"]["n_init"], seed)[0])
    n = len(base)

    # 1. Feature contributions
    rows = []
    for col in base.columns:
        H, p = kruskal(*[base.loc[labels == c, col].to_numpy() for c in range(k)])
        rows.append(dict(feature=col, H=H, p_value=p, epsilon_sq=H / (n - 1)))
    eff = pd.DataFrame(rows).sort_values("epsilon_sq", ascending=False).reset_index(drop=True)
    eff.round(4).to_csv(tables / f"feature_effect_sizes_{args.scope}_k{k}.csv", index=False)
    print("\nFEATURES RANKED BY EFFECT SIZE (epsilon squared, descriptive)\n")
    print(eff[["feature", "epsilon_sq"]].round(3).to_string(index=False))

    prof = base.assign(cluster=labels).groupby("cluster").mean()
    z = (prof - base.mean()) / base.std()
    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(z.T.loc[eff["feature"]], annot=True, fmt=".1f", cmap="coolwarm",
                center=0, ax=ax)
    ax.set_title(f"Cluster signatures (z-scores), {args.scope}, k={k}")
    fig.tight_layout()
    fig.savefig(figs / f"signature_heatmap_{args.scope}_k{k}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(eff["feature"][::-1], eff["epsilon_sq"][::-1])
    ax.set_xlabel("Epsilon squared (Kruskal-Wallis effect size)")
    ax.set_title("Which features separate the clusters most")
    fig.tight_layout()
    fig.savefig(figs / f"feature_effects_{args.scope}_k{k}.png", dpi=130)
    plt.close(fig)

    # 2. Borderline regions
    sil = silhouette_samples(X, labels)
    conf = membership_confidence(X, labels, k, cfg["clustering"]["bootstrap_runs"], 0.8, seed)
    reg = pd.DataFrame({"region_id": base.index, "cluster": labels,
                        "silhouette": sil, "membership_confidence": conf})
    reg["borderline"] = (reg["membership_confidence"] < 0.8) | (reg["silhouette"] < 0.1)
    reg = reg.sort_values("membership_confidence")
    reg.round(3).to_csv(tables / f"region_confidence_{args.scope}_k{k}.csv", index=False)
    print(f"\nBorderline regions: {int(reg['borderline'].sum())} of {n}")
    print("Mean membership confidence by cluster:")
    print(reg.groupby("cluster")["membership_confidence"].mean().round(3).to_string())
    print("\nLOWEST 15 BY MEMBERSHIP CONFIDENCE\n")
    print(reg.head(15).round(3).to_string(index=False))

    # 3. Surrogate decision tree
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=seed)
    cv = cross_val_score(tree, base.to_numpy(), labels, cv=5).mean()
    tree.fit(base.to_numpy(), labels)
    rules = export_text(tree, feature_names=list(base.columns))
    (tables / f"surrogate_rules_{args.scope}_k{k}.txt").write_text(rules, encoding="utf-8")
    print(f"\nSURROGATE TREE (depth 3), 5-fold accuracy: {cv:.2f}\n")
    print(rules)


if __name__ == "__main__":
    main()