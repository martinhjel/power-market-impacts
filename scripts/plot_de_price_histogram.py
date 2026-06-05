"""
Interactive helper for VS Code's Python Interactive window.

Run the cells in order to compare the synthetic DE price distribution in
`data/de_price.parquet` against the actual day-ahead `DE_LU` series.

The full dataset is used for both histograms and summary statistics, while the
displayed x-axis is capped at `PRICE_CEILING` to keep the plot readable.
"""

from __future__ import annotations

# %%
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
REPO_ROOT =  Path.cwd()

PARQUET_PATH = REPO_ROOT / "data" / "de_price.parquet"
CSV_PATH = (
    REPO_ROOT
    / "data"
    / "day_ahead_DE_LU_2024-01-01 00:00:00+01:00_2025-01-01 00:00:00+01:00.csv"
)
OUTPUT_PATH = REPO_ROOT / "images" / "de_price_histogram.pdf"

BINS = 120
PRICE_CEILING = 400.0
PARQUET_COLUMNS: list[str] | None = None
START = None
END = None
ALIGN_OVERLAP = True
SAVE_FIGURE = True
HISTOGRAM_DENSITY = True


def _read_with_fastparquet_direct(path: Path) -> pd.DataFrame:
    from fastparquet import ParquetFile

    pf = ParquetFile(path)
    df = pf.to_pandas(index=False)
    if "__index_level_0__" in df.columns:
        df = df.set_index("__index_level_0__")
    return df


def _read_parquet_with_fallback(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for label in ("default", "pyarrow", "fastparquet"):
        try:
            if label == "default":
                return pd.read_parquet(path)
            if label == "pyarrow":
                return pd.read_parquet(path, engine="pyarrow")
            return _read_with_fastparquet_direct(path)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Failed to read parquet file {path}.\n"
        f"Tried engines:\n" + "\n".join(errors) + "\n"
        "Install fastparquet if pyarrow cannot decode this file:\n"
        "  conda install -c conda-forge fastparquet\n"
        "or\n"
        "  pip install fastparquet"
    )


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.sort_index()
        if out.index.tz is not None:
            out.index = out.index.tz_convert("Europe/Oslo").tz_localize(None)
        return out

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
    out = out.loc[out.index.notna()].sort_index()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("Europe/Oslo").tz_localize(None)
    return out


def _select_columns(df: pd.DataFrame, columns: Iterable[str] | None) -> pd.DataFrame:
    if not columns:
        return df

    requested = [str(col) for col in columns]
    df = df.copy()
    df.columns = df.columns.map(str)
    missing = [col for col in requested if col not in df.columns]
    if missing:
        raise ValueError(f"Requested parquet columns not found: {missing}. Available: {list(df.columns)}")
    return df[requested]


def _flatten_numeric_values(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric columns available for histogram plotting.")

    values = pd.Series(numeric.to_numpy().ravel(), dtype="float64").dropna()
    if values.empty:
        raise ValueError("No non-null numeric values available for histogram plotting.")
    return values


def _format_stats(values: pd.Series) -> str:
    return (
        f"mean={values.mean():.3f}, std={values.std():.3f}, median={values.median():.3f}, "
        f"p10={values.quantile(0.10):.3f}, p90={values.quantile(0.90):.3f}"
    )


# %%
if not PARQUET_PATH.exists():
    raise FileNotFoundError(f"Parquet input not found: {PARQUET_PATH}")
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV input not found: {CSV_PATH}")

df_parquet = _read_parquet_with_fallback(PARQUET_PATH)
df_parquet = _ensure_datetime_index(_select_columns(df_parquet, PARQUET_COLUMNS))

df_csv = pd.read_csv(CSV_PATH, index_col=0)
df_csv = _ensure_datetime_index(df_csv)

if START or END:
    time_slice = slice(START, END)
    df_parquet = df_parquet.loc[time_slice]
    df_csv = df_csv.loc[time_slice]

if ALIGN_OVERLAP:
    overlap_start = max(df_parquet.index.min(), df_csv.index.min())
    overlap_end = min(df_parquet.index.max(), df_csv.index.max())
    df_parquet = df_parquet.loc[overlap_start:overlap_end]
    df_csv = df_csv.loc[overlap_start:overlap_end]

if df_parquet.empty:
    raise ValueError("No parquet data available after applying filters.")
if df_csv.empty:
    raise ValueError("No CSV data available after applying filters.")

parquet_values_raw = _flatten_numeric_values(df_parquet)
csv_values_raw = _flatten_numeric_values(df_csv)

parquet_values = parquet_values_raw
csv_values = csv_values_raw

print(f"Synthetic values: {len(parquet_values):,}")
print(f"Actual values:    {len(csv_values):,}")
print(f"Plot x-axis capped at {PRICE_CEILING:g}")
print("Synthetic:", _format_stats(parquet_values))
print("Actual:   ", _format_stats(csv_values))


# %%
combined_min = min(parquet_values.min(), csv_values.min())
combined_max = max(parquet_values.max(), csv_values.max())
if combined_min == combined_max:
    combined_max = combined_min + 1.0

bin_edges = np.linspace(combined_min, combined_max, BINS + 1)

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(
    parquet_values,
    bins=bin_edges,
    density=HISTOGRAM_DENSITY,
    alpha=0.45,
    color="#1d4ed8",
    label=f"de_price.parquet ({len(parquet_values):,} values)",
)
ax.hist(
    csv_values,
    bins=bin_edges,
    density=HISTOGRAM_DENSITY,
    alpha=0.45,
    color="#dc2626",
    label=f"day_ahead_DE_LU CSV ({len(csv_values):,} values)",
)

ax.axvline(parquet_values.mean(), color="#1d4ed8", linestyle="--", linewidth=1.5)
ax.axvline(csv_values.mean(), color="#dc2626", linestyle="--", linewidth=1.5)
ax.set_xlim(combined_min, min(PRICE_CEILING, combined_max))
ax.set_title(f"DE price distribution with x-axis capped at {PRICE_CEILING:g}")
ax.set_xlabel("Price")
ax.set_ylabel("Density" if HISTOGRAM_DENSITY else "Count")
ax.grid(True, alpha=0.25)
ax.legend()

stats_text = (
    "Synthetic: "
    + _format_stats(parquet_values)
    + "\nActual: "
    + _format_stats(csv_values)
    + f"\nVisible x-axis max: {PRICE_CEILING:g}"
)
ax.text(
    0.99,
    0.99,
    stats_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
)

fig.tight_layout()

if SAVE_FIGURE:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved histogram to {OUTPUT_PATH}")

fig
