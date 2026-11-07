"""First exploratory profiling: coverage, missingness, trajectories."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def short(col):
    return col.replace("_percent_change_from_baseline", "")


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    figs = Path(cfg["paths"]["figures"])
    tables = Path(cfg["paths"]["tables"])
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    cols = cfg["features"]["mobility_columns"]
    scols = [short(c) for c in cols]
    rename = {c: short(c) for c in cols}
    sns.set_theme(style="whitegrid")

    # ------------------------------------------------ NATIONAL
    nat = pd.read_parquet(interim / "mobility_national.parquet").rename(columns=rename)
    print(f"National rows: {len(nat):,}, countries: {nat['country_region'].nunique()}")

    miss = (nat[scols].isna().groupby(nat["country_region"]).mean() * 100).round(1)
    cov = nat.groupby("country_region").agg(
        rows=("date", "size"), first_date=("date", "min"), last_date=("date", "max")
    )
    cov = cov.join(miss)
    cov["avg_missing_pct"] = miss.mean(axis=1).round(1)
    cov = cov.sort_values("avg_missing_pct", ascending=False)
    cov.to_csv(tables / "national_coverage_by_country.csv")
    print("\nTop 15 countries by missing data:\n")
    print(cov.head(15).to_string())

    # Fig 1: missingness heatmap (worst 30 countries)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cov.head(30)[scols], annot=True, fmt=".0f", cmap="Reds",
                cbar_kws={"label": "% missing"}, ax=ax)
    ax.set_title("Missing data by country (30 worst), %")
    ax.set_ylabel("")
    save(fig, figs / "01_national_missingness_heatmap.png")

    # Fig 2: global median trajectory (7-day smoothed)
    daily = (nat.groupby("date")[scols].median()
             .rolling(7, center=True, min_periods=1).mean())
    fig, ax = plt.subplots(figsize=(12, 6))
    daily.plot(ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Median mobility change across countries (7-day smoothed)")
    ax.set_ylabel("% change from baseline")
    save(fig, figs / "02_global_median_trajectories.png")

    # Fig 3: distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, c in zip(axes.ravel(), scols):
        sns.histplot(nat[c].dropna(), bins=60, ax=ax)
        ax.set_title(c)
        ax.set_xlabel("% change")
    fig.suptitle("Distribution of daily national values")
    save(fig, figs / "03_national_distributions.png")

    # Fig 4: correlations between categories
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(nat[scols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation between place categories (national)")
    save(fig, figs / "04_national_correlations.png")

    # ------------------------------------------------ INDIA
    ind = pd.read_parquet(interim / "mobility_india.parquet").rename(columns=rename)
    print("\nIndia rows by level:\n", ind["level"].value_counts().to_string())

    states = ind[ind["level"] == "sub_region_1"]
    s_miss = (states[scols].isna().groupby(states["sub_region_1"]).mean() * 100).round(1)
    s_cov = states.groupby("sub_region_1").agg(rows=("date", "size")).join(s_miss)
    s_cov["avg_missing_pct"] = s_miss.mean(axis=1).round(1)
    s_cov = s_cov.sort_values("avg_missing_pct", ascending=False)
    s_cov.to_csv(tables / "india_state_coverage.csv")
    print(f"\nIndia states/UTs: {len(s_cov)}\n")
    print(s_cov.to_string())

    dist = ind[ind["level"] == "sub_region_2"]
    d_key = dist["sub_region_1"].astype(str) + " | " + dist["sub_region_2"].astype(str)
    d_miss = (dist[scols].isna().groupby(d_key).mean() * 100).round(1)
    d_cov = dist.groupby(d_key).agg(rows=("date", "size")).join(d_miss)
    d_cov["avg_missing_pct"] = d_miss.mean(axis=1).round(1)
    d_cov.to_csv(tables / "india_district_coverage.csv")
    print(f"\nIndia districts: {len(d_cov)}")
    print("Districts with avg missing < 30%:", int((d_cov['avg_missing_pct'] < 30).sum()))

    print("\nProfiling complete. Check outputs/figures and outputs/tables.")


if __name__ == "__main__":
    main()