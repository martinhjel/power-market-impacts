"""Compute capacity factors for the available renewables profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_profiles(path: Path) -> pd.DataFrame:
    """Return the renewables profiles as a DataFrame indexed by timestamp."""
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    return df


def calculate_capacity_factors(df: pd.DataFrame) -> pd.Series:
    """Calculate mean output per profile (capacity factor)."""
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.columns.equals(df.columns):
        missing = set(df.columns) - set(numeric_df.columns)
        raise ValueError(f"Non-numeric columns present in input: {sorted(missing)}")
    return numeric_df.mean().sort_values(ascending=False).rename("capacity_factor")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate capacity factors for renewables profiles.")
    parser.add_argument(
        "--profile-file",
        type=Path,
        default=Path("data/renewables_profiles.parquet"),
        help="Path to the renewables profiles file (parquet or csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the capacity factors as CSV.",
    )
    args = parser.parse_args()

    profiles = load_profiles(args.profile_file)
    capacity_factors = calculate_capacity_factors(profiles)

    if args.output:
        capacity_factors.to_csv(args.output, header=True)
    else:
        print(capacity_factors.to_string())


if __name__ == "__main__":
    main()
