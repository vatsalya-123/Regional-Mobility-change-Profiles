"""Feature engineering: region x period profiles and region-level summaries."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCOPES = ["national", "india_states", "india_districts", "global_states"]


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def short(col):
    return col.replace("_percent_change_from_baseline", "")


def period_features(df, scols, periods, min_cov):
    """Mean, volatility and extreme value per region, category and period."""
    df = df.copy()
    df["is_weekend"] = df["date"].dt.day_of_week >= 5
    region_ids = sorted(df["region_id"].unique())
    frames = []

    for pname, (start, end) in periods.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        n_days = (end - start).days + 1
        sub = df[(df["date"] >= start) & (df["date"] <= end)]
        g = sub.groupby("region_id")
        out = pd.DataFrame(index=region_ids)
        out.index.name = "region_id"

        for c in scols:
            cols = [f"{c}__mean", f"{c}__std", f"{c}__extreme"]
            out[cols[0]] = g[c].mean()
            out[cols[1]] = g[c].std()
            q = 0.95 if c == "residential" else 0.05
            out[cols[2]] = g[c].quantile(q)
            cover = (g[c].count() / n_days).reindex(out.index).fillna(0)
            out.loc[cover < min_cov, cols] = np.nan

        wk = (sub.groupby(["region_id", "is_weekend"])["workplaces"].mean()
              .unstack().reindex(columns=[False, True]))
        out["workplaces__weekend_gap"] = wk[True] - wk[False]

        out["period"] = pname
        frames.append(out.reset_index())

    return pd.concat(frames, ignore_index=True)


def recovery_features(df, comp_cols, first_wave, smooth, band):
    """Trough depth and days to recover for a composite activity index."""
    fw_start, fw_end = (pd.Timestamp(d) for d in first_wave)
    rows = []
    for rid, g in df.groupby("region_id"):
        g = g.sort_values("date")
        comp = g[comp_cols].mean(axis=1)
        comp = comp.rolling(smooth, center=True,
                            min_periods=max(1, smooth // 2)).mean()
        comp.index = pd.DatetimeIndex(g["date"].to_numpy())

        win = comp[(comp.index >= fw_start) & (comp.index <= fw_end)].dropna()
        if win.empty:
            rows.append(dict(region_id=rid, trough_depth=np.nan,
                             recovery_days=np.nan, recovery_censored=np.nan))
            continue

        trough_date, trough = win.idxmin(), win.min()
        after = comp[comp.index >= trough_date].dropna()
        back = after[after >= band]
        if back.empty:
            days, censored = (after.index.max() - trough_date).days, 1
        else:
            days, censored = (back.index[0] - trough_date).days, 0
        rows.append(dict(region_id=rid, trough_depth=trough,
                         recovery_days=days, recovery_censored=censored))
    return pd.DataFrame(rows)


def region_summary(long_df, rec, scols):
    """One compact, interpretable row per region."""
    fw = long_df[long_df["period"] == "first_wave"].set_index("region_id")
    lp = long_df[long_df["period"] == "late_pandemic"].set_index("region_id")
    out = pd.DataFrame(index=fw.index)
    for c in scols:
        out[f"{c}__first_wave_mean"] = fw[f"{c}__mean"]
        out[f"{c}__late_mean"] = lp[f"{c}__mean"]
    out = out.join(rec.set_index("region_id"))
    return out


def main():
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    f = cfg["features"]
    scols = [short(c) for c in f["mobility_columns"]]
    log = []

    for name in SCOPES:
        print(f"\n=== {name} ===")
        df = pd.read_parquet(processed / f"{name}_daily_clean.parquet")

        long_df = period_features(df, scols, cfg["periods"], f["min_period_coverage"])
        feat_cols = [c for c in long_df.columns if "__" in c]
        long_ok = long_df.dropna(subset=feat_cols)

        rec = recovery_features(df, f["composite_columns"], cfg["periods"]["first_wave"],
                                f["smoothing_days"], f["recovery_threshold"])
        summ = region_summary(long_ok, rec, scols)
        summ_ok = summ.dropna(subset=[c for c in summ.columns if c != "recovery_censored"])

        long_ok.to_parquet(processed / f"{name}_features_region_period.parquet", index=False)
        summ_ok.reset_index().to_parquet(processed / f"{name}_features_region.parquet", index=False)
        summ_ok.reset_index().to_csv(processed / f"{name}_features_region.csv", index=False)

        n_regions = df["region_id"].nunique()
        censored = int(summ_ok["recovery_censored"].sum())
        print(f"Regions in daily data     : {n_regions}")
        print(f"Region-period rows kept   : {len(long_ok)} of {len(long_df)}")
        print(f"Region summary rows kept  : {len(summ_ok)} of {n_regions}")
        print(f"Recovery censored (never) : {censored}")
        log.append(dict(scope=name, regions=n_regions,
                        region_period_rows=len(long_ok),
                        summary_rows=len(summ_ok),
                        summary_dropped=n_regions - len(summ_ok),
                        recovery_censored=censored,
                        n_features_summary=summ_ok.shape[1] - 1))

    log_df = pd.DataFrame(log)
    log_df.to_csv(tables / "feature_engineering_log.csv", index=False)
    print("\nFEATURE ENGINEERING LOG\n")
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    main()