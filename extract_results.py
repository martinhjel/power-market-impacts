#!/usr/bin/env python3
"""
Extract frequently used LTM result series to ltm_processed/.

This is a thin compatibility wrapper around scripts/process_ltm_results.py.
It keeps the public repository entrypoint simple while using the same compact
processed_data.parquet format as the paper scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def discover_model_folders(ltm_output: Path) -> list[str]:
    if not ltm_output.exists():
        raise FileNotFoundError(f"LTM output folder not found: {ltm_output}")
    return sorted(path.name for path in ltm_output.iterdir() if path.is_dir())


def run_processing(model_folder: str, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/process_ltm_results.py",
        "--model-folder",
        model_folder,
        "--output-root",
        args.output_root,
        "--reservoir-mode",
        args.reservoir_mode,
        "--workers",
        str(args.workers),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.no_dclines:
        command.append("--no-dclines")
    if args.no_reservoir_aggregates:
        command.append("--no-reservoir-aggregates")

    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract LTM results from ltm_output/ into ltm_processed/processed_data.parquet files."
    )
    parser.add_argument(
        "--model-folder",
        action="append",
        help="Model folder under ltm_output/. Can be repeated. Defaults to every folder under ltm_output/.",
    )
    parser.add_argument("--ltm-output", default="ltm_output", help="Folder containing model result folders.")
    parser.add_argument("--output-root", default="ltm_processed", help="Processed output root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed_data.parquet files.")
    parser.add_argument("--workers", type=int, default=1, help="Number of scenario worker processes.")
    parser.add_argument("--no-dclines", action="store_true", help="Skip DC line flow extraction.")
    parser.add_argument(
        "--reservoir-mode",
        choices=["none", "uprated", "all"],
        default="uprated",
        help="Individual reservoir records to store.",
    )
    parser.add_argument(
        "--no-reservoir-aggregates",
        action="store_true",
        help="Skip area-level aggregate spill/discharge derived from individual reservoirs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_folders = args.model_folder or discover_model_folders(PROJECT_ROOT / args.ltm_output)
    for model_folder in model_folders:
        run_processing(model_folder, args)


if __name__ == "__main__":
    main()
