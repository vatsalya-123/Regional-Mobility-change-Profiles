"""Temporal drift: behavioural states per region-period and transitions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from src.cluster import load_config

MEAN_COLS = ["retail_and_recreation__mean", "grocery_and_pharmacy__mean",
             "transit_stations__mean", "workplaces__mean", "residential__mean"]
ACTIVITY_COLS = ["retail_and_recreation__mean", "transit_stations__mean",
                 "workplaces__mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="national")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    k = args.k

    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    figs = Path(cfg["paths"]["figures"]) / "temporal"
    figs.mkdir(parents=True, exist_ok=True)
    seed = cfg["project"]["random_seed"]
    period_order = list(cfg["periods"].keys())

    df = pd.read_parquet(processed / f"{args.scope}_features_region_period.parquet")
    X = RobustScaler().fit_transform(df[MEAN_COLS].to_numpy())
    print(f"Region-period observations: {len(df)}")

    print("\nSilhouette by k (behavioural states):")
    for kk in range(2, 8):
        lab = KMeans(n_clusters=kk, n_init=20, random_state=seed).fit_predict(X)
        print(f"  k={kk}: {silhouette_score(X, lab):.3f}")

    raw = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
    act = (df.assign(c=raw).groupby("c")[ACTIVITY_COLS].mean().mean(axis=1).sort_values())
    remap = {old: new for new, old in enumerate(act.index)}
    df["state"] = [remap[v] for v in raw]

    profile = df.groupby("state")[MEAN_COLS].mean().round(1)
    profile.insert(0, "n_obs", df["state"].value_counts().sort_index().values)
    profile.to_csv(tables / f"state_profiles_{args.scope}_k{k}.csv")
    print(f"\nSTATE PROFILES (0 = most suppressed activity), k={k}\n")
    print(profile.T.to_string())

    share = (pd.crosstab(df["period"], df["state"], normalize="index")
             .reindex(period_order))
    share.round(3).to_csv(tables / f"state_share_by_period_{args.scope}_k{k}.csv")
    print("\nSHARE OF REGIONS IN EACH STATE, BY PERIOD\n")
    print(share.round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    share.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
    ax.set_ylabel("Share of regions")
    ax.set_title(f"Behavioural states over time, {args.scope} (state 0 = most suppressed)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figs / f"state_share_{args.scope}_k{k}.png", dpi=130)
    plt.close(fig)

    wide = df.pivot(index="region_id", columns="period", values="state").reindex(columns=period_order)
    wide.to_csv(tables / f"state_trajectories_{args.scope}_k{k}.csv")

    pairs = []
    for a, b in zip(period_order[:-1], period_order[1:]):
        t = wide[[a, b]].dropna().astype(int)
        t.columns = ["from", "to"]
        pairs.append(t)
    pairs = pd.concat(pairs)
    counts = (pd.crosstab(pairs["from"], pairs["to"])
              .reindex(index=range(k), columns=range(k), fill_value=0))
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    probs.round(3).to_csv(tables / f"transition_matrix_{args.scope}_k{k}.csv")
    print("\nTRANSITION PROBABILITIES (row = state in one period, column = next period)\n")
    print(probs.round(2).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(probs, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("State in next period")
    ax.set_ylabel("State in this period")
    ax.set_title("Transition probabilities between consecutive periods")
    fig.tight_layout()
    fig.savefig(figs / f"transitions_{args.scope}_k{k}.png", dpi=130)
    plt.close(fig)

    print("\nFIRST-WAVE STATE (rows) vs LATE-PANDEMIC STATE (columns)\n")
    print(pd.crosstab(wide["first_wave"], wide["late_pandemic"]).to_string())

    switches = (wide.diff(axis=1).fillna(0) != 0).sum(axis=1).sort_values(ascending=False)
    switches.rename("n_state_changes").to_csv(tables / f"state_switches_{args.scope}_k{k}.csv")
    print("\nMOST FREQUENT STATE CHANGES\n")
    print(switches.head(10).to_string())


if __name__ == "__main__":
    main()