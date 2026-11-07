
"""
Google Community Mobility Data Loader

Responsibilities:
1. Read the large CSV in chunks.
2. Identify geographic levels.
3. Save Parquet datasets.
4. Generate an initial data-quality report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ID_COLS = [
    "country_region_code",
    "country_region",
    "sub_region_1",
    "sub_region_2",
    "metro_area",
    "iso_3166_2_code",
]

CHUNK_SIZE = 100_000


def load_config(path="config.yaml"):
    """Load project configuration."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def assign_level(df):
    """Identify the geographic level of each record."""

    conditions = [
        df["metro_area"].notna(),
        df["sub_region_2"].notna(),
        df["sub_region_1"].notna(),
    ]

    choices = [
        "metro_area",
        "sub_region_2",
        "sub_region_1",
    ]

    return np.select(
        conditions,
        choices,
        default="national"
    )


def write_parquet(writer_dict, name, df, output_path):
    """Append a DataFrame chunk to a Parquet file."""

    if df.empty:
        return

    table = pa.Table.from_pandas(
        df,
        preserve_index=False
    )

    if name not in writer_dict:
        writer_dict[name] = pq.ParquetWriter(
            output_path,
            table.schema,
            compression="snappy"
        )

    writer_dict[name].write_table(table)


def main():

    cfg = load_config()

    raw_path = Path(cfg["paths"]["raw_data"])
    interim = Path(cfg["paths"]["interim"])
    tables = Path(cfg["paths"]["tables"])

    interim.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    mobility_cols = cfg["features"]["mobility_columns"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {raw_path}"
        )

    # Validate the CSV header.
    header = pd.read_csv(raw_path, nrows=0).columns.tolist()

    missing = [
        col for col in mobility_cols + ["date"]
        if col not in header
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    id_cols = [
        col for col in ID_COLS
        if col in header
    ]

    usecols = id_cols + ["date"] + mobility_cols

    dtypes = {
        col: "string"
        for col in id_cols
    }

    dtypes.update({
        col: "float32"
        for col in mobility_cols
    })

    # Parquet writers.
    writers = {}

    # Data-quality statistics.
    stats = {}

    # Track unique geographic identifiers.
    region_sets = {}

    print("Starting chunk-based data loading...")
    print(f"Dataset: {raw_path}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    try:
        reader = pd.read_csv(
    raw_path,
    usecols=usecols,
    dtype=dtypes,
    parse_dates=["date"],
    chunksize=CHUNK_SIZE,
    keep_default_na=False,
    na_values=[""],)

        for chunk_number, chunk in enumerate(reader, start=1):

            chunk["level"] = assign_level(chunk)

            # Save the complete chunk.
            write_parquet(
                writers,
                "all",
                chunk,
                interim / "mobility_all.parquet"
            )

            # Process each geographic level.
            for level, part in chunk.groupby(
                "level", sort=False
            ):

                write_parquet(
                    writers,
                    level,
                    part,
                    interim / f"mobility_{level}.parquet"
                )

                if level not in stats:
                    stats[level] = {
                        "rows": 0,
                        "date_min": None,
                        "date_max": None,
                        "missing": {
                            col: 0 for col in mobility_cols
                        }
                    }

                    region_sets[level] = set()

                current = stats[level]

                current["rows"] += len(part)

                current["date_min"] = min(
                    current["date_min"],
                    part["date"].min()
                ) if current["date_min"] is not None else part["date"].min()

                current["date_max"] = max(
                    current["date_max"],
                    part["date"].max()
                ) if current["date_max"] is not None else part["date"].max()

                missing_counts = part[mobility_cols].isna().sum()

                for col in mobility_cols:
                    current["missing"][col] += int(
                        missing_counts[col]
                    )

                # Count distinct geographic regions.
                region_columns = {
                    "national": ["country_region_code"],
                    "sub_region_1": [
                        "country_region_code", "sub_region_1"
                    ],
                    "sub_region_2": [
                        "country_region_code",
                        "sub_region_1",
                        "sub_region_2"
                    ],
                    "metro_area": [
                        "country_region_code", "metro_area"
                    ],
                }[level]

                unique_regions = (
                    part[region_columns]
                    .dropna()
                    .drop_duplicates()
                )

                region_sets[level].update(
                    unique_regions.itertuples(
                        index=False, name=None
                    )
                )

            # Save India separately.
            india = chunk[
                chunk["country_region_code"].eq("IN")
            ]

            write_parquet(
                writers,
                "india",
                india,
                interim / "mobility_india.parquet"
            )

            print(
                f"Processed chunk {chunk_number}: "
                f"{len(chunk):,} rows"
            )

    finally:
        for writer in writers.values():
            writer.close()

    # Build data-quality report.
    report_rows = []

    for level, current in stats.items():

        row = {
            "level": level,
            "rows": current["rows"],
            "regions": len(region_sets[level]),
            "date_min": current["date_min"],
            "date_max": current["date_max"],
        }

        for col in mobility_cols:
            missing_pct = (
                current["missing"][col]
                / current["rows"]
            ) * 100

            short_name = col.replace(
                "_percent_change_from_baseline",
                "_missing_pct"
            )

            row[short_name] = round(missing_pct, 2)

        report_rows.append(row)

    report = pd.DataFrame(report_rows)

    report_path = tables / "data_quality_overview.csv"

    report.to_csv(report_path, index=False)

    print("\nDATA LOADING COMPLETED!")
    print("\nData quality overview:\n")
    print(report.to_string(index=False))

    print(f"\nQuality report saved: {report_path}")
    print(f"Parquet files saved in: {interim}")


if __name__ == "__main__":
    main()
