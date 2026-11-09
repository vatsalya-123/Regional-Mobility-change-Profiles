"""Hold-out validation: do clusters found on a training split reappear on unseen regions?"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.cluster import FEATURE_SETS, SCALERS, load_config, load_features


def prediction_strength(assigned, test_labels, k):
    """Tibshirani and Walther: for each test cluster, the share of point pairs that the
    training centroids also keep together. The minimum over clusters is returned."""
    strengths = []
    for j in range(k):
        members = assigned[test_labels == j]
        m = len(members)
        if m < 2:
            continue
        counts = np.bincount(members, minlength=k)
        strengths.append((counts * (counts - 1)).sum() / (m * (m - 1)))
    return min(strengths) if strengths else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="national")
    ap.add_argument("--feature_set", default="no_parks", choices=FEATURE_SETS)
    ap.add_argument("--scaler", default="robust", choices=list(SCALERS))
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--n_rep", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    figs = Path(cfg["paths"]["figures"]) / "evaluation"
    figs.mkdir(parents=True, exist_ok=True)

    base = load_features(processed / f"{args.scope}_features_region.parquet", args.feature_set)
    X_raw = base.to_numpy()
    n = len(X_raw)
    n_test = int(round(args.test_frac * n))
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    k_values = list(range(2, 9))
    rows = []

    for rep in range(args.n_rep):
        perm = rng.permutation(n)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        scaler = SCALERS[args.scaler]().fit(X_raw[train_idx])  # fit on training data only
        X_tr = scaler.transform(X_raw[train_idx])
        X_te = scaler.transform(X_raw[test_idx])
        for k in k_values:
            km_tr = KMeans(n_clusters=k, n_init=10,
                           random_state=int(rng.integers(1_000_000_000))).fit(X_tr)
            km_te = KMeans(n_clusters=k, n_init=10,
                           random_state=int(rng.integers(1_000_000_000))).fit(X_te)
            assigned = km_tr.predict(X_te)
            rows.append(dict(
                rep=rep, k=k,
                prediction_strength=prediction_strength(assigned, km_te.labels_, k),
                ari=adjusted_rand_score(km_te.labels_, assigned)))

    res = pd.DataFrame(rows)
    summ = res.groupby("k").agg(
        ps_mean=("prediction_strength", "mean"),
        ps_p10=("prediction_strength", lambda s: s.quantile(0.10)),
        ari_mean=("ari", "mean"),
        ari_p10=("ari", lambda s: s.quantile(0.10)),
    ).round(3)
    summ.to_csv(tables / f"holdout_{args.scope}.csv")

    print(f"\nHOLD-OUT VALIDATION ({args.scope}, {n - n_test} train / {n_test} test, "
          f"{args.n_rep} random splits, {args.feature_set}, {args.scaler})\n")
    print(summ.to_string())
    ok = summ[summ["ps_mean"] >= 0.8]
    print("\nLargest k with mean prediction strength >= 0.8:",
          int(ok.index.max()) if len(ok) else "none")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(summ.index, summ["ps_mean"], marker="o", label="mean")
    axes[0].plot(summ.index, summ["ps_p10"], marker="o", linestyle="--", label="10th percentile")
    axes[0].axhline(0.8, color="grey", linestyle=":")
    axes[0].set_title("Prediction strength (0.8 is a common benchmark)")
    axes[1].plot(summ.index, summ["ari_mean"], marker="o", label="mean")
    axes[1].plot(summ.index, summ["ari_p10"], marker="o", linestyle="--", label="10th percentile")
    axes[1].set_title("Agreement on unseen regions (ARI)")
    for ax in axes:
        ax.set_xlabel("k")
        ax.legend()
    fig.suptitle(f"Hold-out validation, {args.scope}")
    fig.tight_layout()
    fig.savefig(figs / f"holdout_{args.scope}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()