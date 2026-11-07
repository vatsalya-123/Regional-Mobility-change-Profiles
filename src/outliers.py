"""Outlier diagnostic: robust z-scores of region profile features."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.cluster import SCOPES, load_config


def main():
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    all_rows = []

    for scope in SCOPES:
        df = pd.read_parquet(processed / f"{scope}_features_region.parquet").set_index("region_id")
        df = df.drop(columns=["recovery_censored"])
        med = df.median()
        mad = ((df - med).abs().median() * 1.4826).replace(0, np.nan)
        z = ((df - med) / mad).abs()
        worst = z.idxmax(axis=1)
        out = pd.DataFrame({
            "scope": scope,
            "region_id": df.index,
            "max_robust_z": z.max(axis=1).to_numpy(),
            "driving_feature": worst.to_numpy(),
            "value": [df.loc[r, f] for r, f in zip(df.index, worst)],
        }).sort_values("max_robust_z", ascending=False)
        all_rows.append(out)

        print(f"\n=== {scope} (n={len(out)}) ===")
        print(out.head(8).round(1).to_string(index=False))
        print("Regions above robust-z threshold:",
              {t: int((out["max_robust_z"] > t).sum()) for t in (8, 10, 15, 20)})

    pd.concat(all_rows).round(2).to_csv(tables / "outlier_diagnostics.csv", index=False)
    print("\nSaved outlier_diagnostics.csv")


if __name__ == "__main__":
    main()