from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import load_scenarios, logger
from scripts.processed_results import processed_data_path_for_result

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
AREA_PREFIXES = ("NO", "SE", "DK", "FI", "EE", "LT", "SNII", "VVD", "UN")

# SCENARIOS = {
#     # "N-LLPS+": "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
#     # "OWN-LLPS+": "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
#     # "OW-LLPS+": "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
#     # "N-BA+": "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
#     # "OWN-BA+": "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
#     # "OW-BA+": "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
#     # "N-LLPS": "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
#     # "OWN-LLPS": "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
#     # "OW-LLPS": "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
#     # "N-BA": "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
#     # "OWN-BA": "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
#     # "OW-BA": "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
#     # "B+": "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF",
#     # "B": "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
#     # "B30": "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF",
#     # "SMR300-BA": "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
#     # "SMR300-LLPS": "SMR300LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
#     # "SMR600-BA": "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
#     # "SMR600-LLPS": "SMR600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
#     # "SMR900-BA": "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
#     # "SMR900-LLPS": "SMR900LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
#     # "SMR1200-BA": "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
#     # "SMR1200-LLPS": "SMR1200LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
#     # "SMR1600-BA": "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
#     # "SMR1600-LLPS": "SMR1600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
#     # "LMR2000-BA": "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
#     # "LMR2000-LLPS": "LMR2000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
#     # "LMR3000-BA": "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
#     # "LMR3000-LLPS": "LMR3000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
#     # "LMR4000-BA": "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
#     # "LMR4000-LLPS": "LMR4000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
# }


def discover_scenario_paths(model_folder: str, scenario_names: List[str] | None = None) -> dict[str, Path]:
    model_root = PROJECT_ROOT / "ltm_output" / model_folder
    if not model_root.exists():
        raise FileNotFoundError(f"Model folder not found: {model_root}")

    if scenario_names:
        scenario_paths = {name: model_root / name for name in scenario_names}
    else:
        scenario_paths = {
            path.name: path
            for path in sorted(model_root.iterdir())
            if path.is_dir() and (path / "run_folder" / "emps").exists()
        }

    if not scenario_paths:
        raise RuntimeError(f"No scenario folders found under {model_root}")

    missing = [name for name, path in scenario_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Scenario folder(s) not found under {model_root}: {', '.join(missing)}")

    missing_processed = [
        str(processed_data_path_for_result(path))
        for path in scenario_paths.values()
        if not processed_data_path_for_result(path).exists()
    ]
    if missing_processed:
        raise FileNotFoundError(
            "Processed result data is required before market dispatch export. Missing:\n"
            + "\n".join(missing_processed)
        )

    return scenario_paths


def _align(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(index=base.index, columns=base.columns, fill_value=0.0)


def _parse_dcline_name(name: str) -> Tuple[str | None, str | None]:
    parts = name.split("_")
    if len(parts) < 3:
        return None, None
    node_a = parts[1]
    node_b = "_".join(parts[2:])
    return node_a, node_b


def _net_import_export(scenario, busbar: str, base: pd.DataFrame) -> pd.DataFrame:
    net_ie = pd.DataFrame(0.0, index=base.index, columns=base.columns)
    for line_name in scenario.get_dcline_names():
        node_a, node_b = _parse_dcline_name(line_name)
        if node_a is None:
            continue
        if node_a != busbar and node_b != busbar:
            continue
        df_line = scenario.get_dcline_flow(line_name)
        df_line = _align(df_line, base)
        if node_a == busbar:
            net_ie += -df_line
        else:
            net_ie += df_line
    return net_ie


def _to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    long_df = df.stack().rename(value_name).reset_index()
    long_df.columns = ["timestamp", "scenario", value_name]
    return long_df


def _is_nordic_area(name: str) -> bool:
    return name.startswith(AREA_PREFIXES)


def _get_processed_or_zero(scenario, area: str, getter_name: str, base: pd.DataFrame) -> pd.DataFrame:
    try:
        return getattr(scenario, getter_name)(area)
    except KeyError:
        return pd.DataFrame(0.0, index=base.index, columns=base.columns)


def _get_nuclear_or_zero(scenario, area: str, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = _get_processed_or_zero(scenario, area, "get_fixed_nuclear_for_busbar", base)
    try:
        total = scenario.get_total_nuclear_for_busbar(area)
    except KeyError:
        total = fixed
    return fixed, total


def build_market_dispatch(scenario, areas: List[str] | None) -> pd.DataFrame:
    busbars = scenario.get_busbar_names()
    if areas:
        busbars = [area for area in busbars if area in areas]
    else:
        busbars = [area for area in busbars if _is_nordic_area(area)]

    all_rows = []
    for area_name in busbars:
        df_load = scenario.get_load_for_busbar(area_name)
        df_hydro = _get_processed_or_zero(scenario, area_name, "get_hydro_production_for_busbar", df_load)
        df_market_steps = _get_processed_or_zero(scenario, area_name, "get_market_steps_for_busbar", df_load)
        df_solar = _get_processed_or_zero(scenario, area_name, "get_solar_for_busbar", df_load)
        df_wind_on = _get_processed_or_zero(scenario, area_name, "get_onshore_wind_for_busbar", df_load)
        df_wind_off = _get_processed_or_zero(scenario, area_name, "get_offshore_wind_for_busbar", df_load)
        df_fixed_nuclear, df_nuclear = _get_nuclear_or_zero(scenario, area_name, df_load)
        df_flexible_nuclear = df_nuclear.sub(df_fixed_nuclear, fill_value=0.0).clip(lower=0.0)
        df_market_steps = df_market_steps.sub(df_flexible_nuclear, fill_value=0.0)
        df_net_ie = _net_import_export(scenario, area_name, df_load)

        df_hydro = _align(df_hydro, df_load)
        df_market_steps = _align(df_market_steps, df_load)
        df_solar = _align(df_solar, df_load)
        df_wind_on = _align(df_wind_on, df_load)
        df_wind_off = _align(df_wind_off, df_load)
        df_nuclear = _align(df_nuclear, df_load)

        data_frames = [
            _to_long(df_load, "load"),
            _to_long(df_hydro, "hydro"),
            _to_long(df_wind_on, "onshore_wind"),
            _to_long(df_wind_off, "offshore_wind"),
            _to_long(df_nuclear, "nuclear"),
            _to_long(df_solar, "solar"),
            _to_long(df_market_steps, "market_steps"),
            _to_long(df_net_ie, "net_import_export"),
        ]

        merged = data_frames[0]
        for df in data_frames[1:]:
            merged = merged.merge(df, on=["timestamp", "scenario"], how="left")

        merged["area"] = area_name
        all_rows.append(merged)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def write_market_dispatch(scenario, df: pd.DataFrame) -> Path:
    processed_dir = processed_data_path_for_result(scenario.result_path).parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "market_dispatch.pkl"
    df.to_pickle(output_path)
    logger.info(f"Wrote to {output_path}")
    return output_path


def _run_single(scenario_label: str, scenario) -> None:
    logger.info(f"Processing scenario: {scenario_label} ({scenario.name})")
    df = build_market_dispatch(scenario, areas=None)
    if df.empty:
        logger.warning(f"Skipping {scenario.name}: no data produced.")
        return

    df = df.set_index(["area", "scenario", "timestamp"])
    diff = df["load"] - df.iloc[:, 1:].sum(axis=1)

    logger.info(f"Max diff: {diff.max()}, std: {diff.std()}, mean: {diff.mean()}")
    if diff.abs().max() > 30:
        logger.warning(
            f"Large difference between load and sum of components for {scenario.name}. Max diff: {diff.max()}"
        )

    logger.info("Writing results to file")
    write_market_dispatch(scenario, df)


def main(
    parallel: bool = False,
    max_workers: int | None = None,
    model_folder: str = MODEL_FOLDER,
    scenario_names: List[str] | None = None,
) -> None:
    scenario_paths = discover_scenario_paths(model_folder, scenario_names)
    scenarios = load_scenarios(scenario_paths)

    if parallel:
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single, label, scenario): label for label, scenario in scenarios.items()}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(label)
                    logger.exception("Scenario %s failed: %s", label, exc)
        if failures:
            raise RuntimeError(f"{len(failures)} scenario(s) failed: {', '.join(failures)}")
    else:
        for scenario_label, scenario in scenarios.items():
            _run_single(scenario_label, scenario)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export market dispatch results.")
    parser.add_argument("--model-folder", default=MODEL_FOLDER, help="Folder under ltm_output to process.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario folder names to process. Defaults to all scenario folders in the model folder.",
    )
    parser.add_argument("--parallel", action="store_true", help="Run scenarios in parallel.")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers for parallel runs.")
    args = parser.parse_args()
    main(
        parallel=args.parallel,
        max_workers=args.workers,
        model_folder=args.model_folder,
        scenario_names=args.scenarios,
    )
