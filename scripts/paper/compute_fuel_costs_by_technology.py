#!/usr/bin/env python3
"""
Summarize fuel cost values by technology type.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _format_unique(values: list[float]) -> str:
    return ";".join(f"{value:g}" for value in values)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "type" not in df.columns:
        raise KeyError("Missing required column: type")
    if "fuelcost" not in df.columns:
        raise KeyError("Missing required column: fuelcost")

    working = df[["type", "fuelcost"]].copy()
    working["fuelcost"] = pd.to_numeric(working["fuelcost"], errors="coerce")

    grouped = working.groupby("type", dropna=False)
    summary = grouped["fuelcost"].agg(
        count_units="size",
        count_fuelcost="count",
        fuelcost_min="min",
        fuelcost_max="max",
        fuelcost_mean="mean",
        fuelcost_median="median",
    )
    unique_costs = grouped["fuelcost"].apply(lambda series: sorted(set(series.dropna().tolist())))
    summary["unique_fuelcosts"] = unique_costs.apply(_format_unique)
    return summary.reset_index()


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    display = summary.copy()
    display = display.rename(
        columns={
            "type": "Technology",
            "count_units": "Units",
            "count_fuelcost": "Units with fuel cost",
            "fuelcost_min": "Min",
            "fuelcost_max": "Max",
            "fuelcost_mean": "Mean",
            "fuelcost_median": "Median",
            "unique_fuelcosts": "Unique values",
        }
    )
    latex = display.to_latex(
        index=False,
        escape=True,
        float_format="%.2f",
        caption="Fuel cost values by technology in the PowerGAMA input data.",
        label="tab:fuel_costs",
    )
    path.write_text(latex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fuel cost values by technology type.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.cwd() / "data/NordicNuclearAnalysis/CASE_2025/scenario_BM/data/system/combined/generator_BM_v100.csv",
        help="Generator CSV with type and fuelcost columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "visualizations" / "paper",
        help="Directory for fuel_costs_by_technology.csv and fuel_costs_by_technology.tex.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    summary = build_summary(df)

    csv_path = args.output_dir / "fuel_costs_by_technology.csv"
    tex_path = args.output_dir / "fuel_costs_by_technology.tex"
    summary.to_csv(csv_path, index=False)
    write_latex_table(summary, tex_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
