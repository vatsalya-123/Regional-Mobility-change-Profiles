"""Sanity tests for the preprocessed datasets."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

with open("config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

PROCESSED = Path(CFG["paths"]["processed"])
THRESHOLD = CFG["preprocessing"]["max_missing_fraction"]
FEATURES = [c.replace("_percent_change_from_baseline", "")
            for c in CFG["features"]["mobility_columns"]]
SCOPES = ["national", "india_states", "india_districts", "global_states"]


@pytest.fixture(scope="module", params=SCOPES)
def scope_data(request):
    name = request.param
    df = pd.read_parquet(PROCESSED / f"{name}_daily_clean.parquet")
    meta = pd.read_csv(PROCESSED / f"{name}_region_meta.csv", keep_default_na=False)
    return name, df, meta


def test_no_duplicate_region_days(scope_data):
    _, df, _ = scope_data
    assert not df.duplicated(["region_id", "date"]).any()


def test_complete_daily_grid(scope_data):
    _, df, _ = scope_data
    expected = (df["date"].max() - df["date"].min()).days + 1
    assert (df.groupby("region_id").size() == expected).all()


def test_meta_matches_data(scope_data):
    _, df, meta = scope_data
    assert set(meta["region_id"]) == set(df["region_id"])


def test_missing_within_threshold(scope_data):
    _, df, _ = scope_data
    frac = df[FEATURES].isna().groupby(df["region_id"]).mean()
    assert (frac <= THRESHOLD).all().all()


def test_values_in_plausible_range(scope_data):
    _, df, _ = scope_data
    assert df[FEATURES].min().min() >= -100


def test_known_regions():
    nat = pd.read_csv(PROCESSED / "national_region_meta.csv", keep_default_na=False)
    names = set(nat["country_region"])
    assert "India" in names and "Namibia" in names
    assert "Liechtenstein" not in names