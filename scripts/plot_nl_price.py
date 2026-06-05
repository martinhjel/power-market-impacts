"""
Quick helper to read and plot the Dutch day-ahead price profile in `data/nl_price.parquet`.

The script saves a PNG (and optionally shows the plot) so you can quickly inspect the
profile without opening notebooks.

Usage:
    python scripts/plot_nl_price.py
    python scripts/plot_nl_price.py --start 2024-06-01 --end 2024-06-07 --output images/nl_price_week.png
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Try to guarantee a DatetimeIndex for plotting."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # If there is a likely timestamp column, use it.
    for candidate in ("timestamp", "time", "date", "datetime"):
        if candidate in df.columns:
            idx = pd.to_datetime(df[candidate], errors="coerce", utc=True)
            df = df.drop(columns=[candidate])
            df.index = idx
            return df

    # Fallback: attempt to coerce the existing index.
    df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    return df


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Plot Dutch day-ahead power prices from nl_price.parquet")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "nl_price.parquet",
        help="Path to the nl_price.parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "images" / "nl_price.png",
        help="Where to save the plot image (PNG).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start datetime (e.g. 2024-01-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end datetime (e.g. 2025-01-01).",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Optional list of columns to plot. Defaults to all columns.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the plot window (in addition to saving).",
    )
    return parser


def _select_columns(df: pd.DataFrame, columns: Iterable[str] | None) -> pd.DataFrame:
    if not columns:
        return df

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Requested columns not found: {missing}. Available: {list(df.columns)}")

    return df[list(columns)]


def main() -> None:
    args = _parse_args().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    df = pd.read_parquet(args.input)
    df = _ensure_datetime_index(df)

    # Drop rows with invalid/NaT index entries.
    df = df.loc[df.index.notna()]
    df = _select_columns(df, args.columns)

    if args.start or args.end:
        df = df.loc[slice(args.start, args.end)]

    if df.empty:
        raise ValueError("No data available to plot after applying filters.")

    # Matplotlib cannot handle tz-aware indexes in some backends; strip TZ for plotting.
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    fig, ax = plt.subplots(figsize=(12, 4))
    df.plot(ax=ax, linewidth=1.0)
    ax.set_title("Dutch day-ahead price profile (NL)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
