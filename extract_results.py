#!/usr/bin/env python3
"""
Extract relevant data from LTM simulation results into compact parquet files.

Run this script once from the repo root to populate the processed/ folder from
ltm_output/. After extraction, all figure and table scripts work without
ltm_output/ and without the lpr_sintef_bifrost library.

Requires: lpr_sintef_bifrost (the simulation library), pandas, pyarrow

Usage:
    python extract_results.py
    python extract_results.py --ltm-output /path/to/ltm_output --processed processed
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Areas to extract hydro and reservoir data for.
# Price and load are extracted for all busbars found in the model.
HYDRO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]


def _save(df, path: Path) -> None:
    df.to_parquet(path)


def extract_scenario(scenario_path: Path, out_path: Path) -> None:
    """Extract all relevant data from one scenario folder to parquet files."""
    from lpr_sintef_bifrost.ltm import LTM
    from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

    session = LTM.session_from_folder(scenario_path / "run_folder/emps")
    model = session.model
    busbars = {b.name: b for b in model.busbars()}

    metadata: dict = {"busbars": {}}

    for area, busbar in busbars.items():
        area_path = out_path / area
        area_path.mkdir(parents=True, exist_ok=True)

        # Price — all areas
        try:
            _save(df_from_pyltm_result(busbar.market_result_price()), area_path / "price.parquet")
        except Exception as e:
            logger.warning(f"  {area}: price failed: {e}")

        # Load — all areas
        try:
            _save(df_from_pyltm_result(busbar.sum_load()), area_path / "load.parquet")
        except Exception as e:
            logger.warning(f"  {area}: load failed: {e}")

        rsv_names: list[str] = []

        if area in HYDRO_AREAS:
            # Aggregate hydro production
            try:
                _save(df_from_pyltm_result(busbar.sum_hydro_production()), area_path / "hydro_production.parquet")
            except Exception as e:
                logger.warning(f"  {area}: hydro_production failed: {e}")

            # Aggregate reservoir
            try:
                _save(df_from_pyltm_result(busbar.sum_reservoir()), area_path / "reservoir_agg.parquet")
            except Exception as e:
                logger.warning(f"  {area}: reservoir_agg failed: {e}")

            # Individual reservoirs
            for rsv in busbar.reservoirs():
                rsv_name = rsv.name
                rsv_path = area_path / "reservoirs" / rsv_name
                rsv_path.mkdir(parents=True, exist_ok=True)
                rsv_names.append(rsv_name)

                for attr, fname in [
                    ("reservoir", "content.parquet"),
                    ("spill", "spill.parquet"),
                    ("discharge", "discharge.parquet"),
                    ("production", "production.parquet"),
                ]:
                    try:
                        _save(df_from_pyltm_result(getattr(rsv, attr)(time_axis=True)), rsv_path / fname)
                    except Exception as e:
                        logger.warning(f"  {area}/{rsv_name}: {attr} failed: {e}")

        metadata["busbars"][area] = {"reservoirs": rsv_names}

    # Copy ltm_model.json — needed by calculate_hydro_uprate_value_factor.py
    model_json = scenario_path / "run_folder/emps/ltm_model.json"
    if model_json.exists():
        shutil.copy2(model_json, out_path / "ltm_model.json")

    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def extract_model_folder(ltm_model_path: Path, processed_model_path: Path) -> None:
    scenario_dirs = [
        d for d in sorted(ltm_model_path.iterdir())
        if d.is_dir() and (d / "run_folder/emps").exists()
    ]
    logger.info(f"Found {len(scenario_dirs)} scenarios in {ltm_model_path.name}")

    for scenario_dir in scenario_dirs:
        out_path = processed_model_path / scenario_dir.name

        if (out_path / "metadata.json").exists():
            logger.info(f"  Skipping {scenario_dir.name} (already extracted)")
            continue

        logger.info(f"  Extracting {scenario_dir.name} ...")
        out_path.mkdir(parents=True, exist_ok=True)
        try:
            extract_scenario(scenario_dir, out_path)
            logger.info(f"  Done: {scenario_dir.name}")
        except Exception as e:
            logger.error(f"  Failed {scenario_dir.name}: {e}")
            shutil.rmtree(out_path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LTM results to parquet.")
    parser.add_argument("--ltm-output", default="ltm_output", help="Path to ltm_output/ folder")
    parser.add_argument("--processed", default="processed", help="Output folder for extracted data")
    args = parser.parse_args()

    ltm_output = Path(args.ltm_output)
    processed = Path(args.processed)

    if not ltm_output.exists():
        logger.error(f"ltm_output folder not found: {ltm_output}")
        return

    for model_folder in sorted(ltm_output.iterdir()):
        if not model_folder.is_dir():
            continue
        logger.info(f"\nProcessing model folder: {model_folder.name}")
        extract_model_folder(model_folder, processed / model_folder.name)

    logger.info("\nExtraction complete.")


if __name__ == "__main__":
    main()
