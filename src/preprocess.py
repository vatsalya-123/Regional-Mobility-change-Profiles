"""Preprocessing: scope selection, region filtering, gap handling."""
from pathlib import Path

import pandas as pd
import yaml

SCOPES = {
    "national": dict(file="mobility_national.parquet", level="national",
                     keys=["country_region"]),
    "india_states": dict(file="mobility_india.parquet", level="sub_region_1",
                         keys=["country_region", "sub_region_1"]),
    "india_districts": dict(file="mobility_india.parquet", level="sub_region_2",
                            keys=["country_region", "sub_region_1", "sub_region_2"]),
    "global_states": dict(file="mobility_sub_region_1.parquet", level="sub_region_1",
                          keys=["country_region", "sub_region_1"]),
}


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def short(col):
    return col.replace("_percent_change_from_baseline", "")


def interpolate_short_gaps(s, max_gap):
    """Linear interpolation inside the series; gaps longer than max_gap stay NaN."""
    filled = s.interpolate(method="linear", limit_area="inside")
    is_na = s.isna()
    block = (is_na != is_na.shift()).cumsum()
    gap_len = is_na.groupby(block).transform("sum")
    return filled.mask(is_na & (gap_len > max_gap))


def fill_series(s, max_gap, edge_days):
    """Short-gap interpolation plus limited fill at the start and end."""
    out = interpolate_short_gaps(s, max_gap)
    valid = out.notna().to_numpy()
    if not valid.any():
        return out
    first = int(valid.argmax())
    last = len(valid) - 1 - int(valid[::-1].argmax())
    if first > 0:
        n = min(first, edge_days)
        out.iloc[first - n:first] = out.iloc[first]
    if last < len(out) - 1:
        n = min(len(out) - 1 - last, edge_days)
        out.iloc[last + 1:last + 1 + n] = out.iloc[last]
    return out


def run_scope(name, spec, cfg, rename, scols):
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    pp = cfg["preprocessing"]
    thr = pp["max_missing_fraction"]

    print(f"\n=== {name} ===")
    df = pd.read_parquet(interim / spec["file"],
                         filters=[("level", "==", spec["level"])])
    df = df.rename(columns=rename)

    keys = spec["keys"]
    df["region_id"] = df[keys[0]].astype(str)
    for k in keys[1:]:
        df["region_id"] = df["region_id"] + " | " + df[k].astype(str)

    raw_rows = len(df)
    df = df.drop_duplicates(["region_id", "date"])
    dupes = raw_rows - len(df)

    meta_cols = keys + [c for c in ("country_region_code", "iso_3166_2_code")
                        if c in df.columns]
    meta = df.groupby("region_id")[meta_cols].first()

    # Full daily grid: absent days become explicit NaN rows
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    ids = sorted(df["region_id"].unique())
    idx = pd.MultiIndex.from_product([ids, full_dates], names=["region_id", "date"])
    grid = df.set_index(["region_id", "date"])[scols].reindex(idx)

    # Drop regions with any feature above the missing threshold
    miss = grid.isna().groupby(level="region_id").mean()
    bad = (miss > thr).any(axis=1)
    miss[bad].round(3).to_csv(tables / f"dropped_regions_{name}.csv")
    keep_ids = miss.index[~bad]
    grid = grid[grid.index.get_level_values("region_id").isin(keep_ids)]

    cells = grid.size
    na_before = int(grid.isna().sum().sum())

    # Fill short gaps only
    grid = grid.reset_index()
    for c in scols:
        grid[c] = grid.groupby("region_id")[c].transform(
            fill_series, max_gap=pp["max_gap_days"], edge_days=pp["edge_fill_days"]
        )
    na_after = int(grid[scols].isna().sum().sum())

    grid.to_parquet(processed / f"{name}_daily_clean.parquet", index=False)
    meta.loc[keep_ids].reset_index().to_csv(
        processed / f"{name}_region_meta.csv", index=False
    )

    print(f"Regions: {len(ids)} raw -> {len(keep_ids)} kept, {int(bad.sum())} dropped")
    if bad.any():
        print("Dropped:", list(miss.index[bad])[:15])
    print(f"Missing cells: {na_before / cells:.2%} before fill -> {na_after / cells:.2%} after")

    return dict(
        scope=name, raw_rows=raw_rows, duplicates_removed=dupes,
        days=len(full_dates), regions_raw=len(ids), regions_dropped=int(bad.sum()),
        regions_kept=len(keep_ids), threshold=thr,
        max_gap_days=pp["max_gap_days"],
        missing_pct_before_fill=round(na_before / cells * 100, 2),
        missing_pct_after_fill=round(na_after / cells * 100, 2),
    )


def main():
    cfg = load_config()
    cols = cfg["features"]["mobility_columns"]
    rename = {c: short(c) for c in cols}
    scols = [short(c) for c in cols]
    Path(cfg["paths"]["processed"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["tables"]).mkdir(parents=True, exist_ok=True)

    log = [run_scope(n, s, cfg, rename, scols) for n, s in SCOPES.items()]
    log_df = pd.DataFrame(log)
    log_df.to_csv(Path(cfg["paths"]["tables"]) / "preprocessing_log.csv", index=False)
    print("\nPREPROCESSING LOG\n")
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    main()