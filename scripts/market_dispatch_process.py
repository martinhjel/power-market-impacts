from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Iterable, List

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import ScenarioResults, load_scenarios, logger
from scripts.processed_results import processed_data_path_for_result

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
AREA_PREFIXES = ("NO", "SE", "DK", "FI", "EE", "LT", "SNII", "VVD", "UN")
GENERATOR_CSV = Path.cwd() / "data/NordicNuclearAnalysis/CASE_2025/scenario_BM/data/system/combined/generator_BM_v100.csv"
FUELCOST_TOL = 0.5

SCENARIOS = {
    # "N-LLPS+": "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    # "OWN-LLPS+": "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    # "OW-LLPS+": "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    # "N-BA+": "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    # "OWN-BA+": "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    # "OW-BA+": "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    # "N-LLPS": "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    # "OWN-LLPS": "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    # "OW-LLPS": "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    # "N-BA": "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    # "OWN-BA": "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    # "OW-BA": "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "B+": "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF",
    # "B": "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
    # "B30": "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF",
    # "SMR300-BA": "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    # "SMR300-LLPS": "SMR300LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    # "SMR600-BA": "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    # "SMR600-LLPS": "SMR600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    # "SMR900-BA": "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    # "SMR900-LLPS": "SMR900LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    # "SMR1200-BA": "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    # "SMR1200-LLPS": "SMR1200LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    # "SMR1600-BA": "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    # "SMR1600-LLPS": "SMR1600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    # "LMR2000-BA": "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    # "LMR2000-LLPS": "LMR2000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    # "LMR3000-BA": "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    # "LMR3000-LLPS": "LMR3000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    # "LMR4000-BA": "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
    # "LMR4000-LLPS": "LMR4000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
}


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
            "Processed result data is required before market dispatch processing. Missing:\n"
            + "\n".join(missing_processed)
        )

    return scenario_paths


def read_ltm_model(result_path: Path) -> dict:
    model_path = result_path / "run_folder" / "emps" / "ltm_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"LTM model file not found: {model_path}")

    with open(model_path) as f:
        payload = json.load(f)

    return payload.get("model", {})



def normalize_technology_name(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return "unknown"
        if len(value) == 1:
            return str(value[0])
        return "+".join(str(v) for v in value)
    if value is None:
        return "unknown"
    if isinstance(value, float) and np.isnan(value):
        return "unknown"
    return str(value)


def sum_capacities(values: Iterable[object]) -> object:
    total = None
    for value in values:
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = float(value.ravel()[0])
            else:
                value = pd.Series(value.ravel())
        if total is None:
            total = value
            continue
        if isinstance(total, pd.Series) and isinstance(value, pd.Series):
            total = total.add(value, fill_value=0.0)
        elif isinstance(total, pd.Series):
            total = total + float(value)
        elif isinstance(value, pd.Series):
            total = value + float(total)
        else:
            total = float(total) + float(value)
    return total if total is not None else 0.0


def market_step_busbars_from_model(model: dict) -> dict[str, str]:
    market_step_names = {str(item.get("name")) for item in model.get("market_steps", []) if item.get("name")}
    busbar_names = {str(item.get("name")) for item in model.get("busbars", []) if item.get("name")}
    busbars_by_name: dict[str, str] = {}
    for connection in model.get("connections", []):
        from_name = str(connection.get("from", ""))
        to_name = str(connection.get("to", ""))
        if from_name in market_step_names and to_name in busbar_names:
            busbars_by_name[from_name] = to_name
        elif to_name in market_step_names and from_name in busbar_names:
            busbars_by_name[to_name] = from_name
    return busbars_by_name


def _read_timeseries_payload(payload: dict, run_folder: Path) -> tuple[np.ndarray, np.ndarray]:
    if not payload:
        return np.array([]), np.array([])

    external_reference = payload.get("external_reference")
    if external_reference:
        h5_path = run_folder / external_reference.get("filename", "input.h5")
        group_path = external_reference["path"]
        with h5py.File(h5_path, "r") as h5:
            group = h5[group_path]
            return np.asarray(group["vals"]), np.asarray(group["times"])

    return np.asarray(payload.get("scenarios", [])), np.asarray(payload.get("timestamps", []))


def _timestamps_to_index(timestamps: np.ndarray) -> pd.DatetimeIndex:
    if np.issubdtype(timestamps.dtype, np.number):
        return pd.to_datetime(timestamps, unit="ms")

    index = pd.to_datetime(timestamps)
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        index = index.tz_convert(None)
    return index


def _timeseries_first_value(payload: dict, run_folder: Path) -> float | None:
    values, _ = _read_timeseries_payload(payload, run_folder)
    if values.size == 0:
        return None
    try:
        return float(values.ravel()[0])
    except (ValueError, TypeError):
        return None


def _timeseries_capacity(payload: dict, run_folder: Path) -> float | pd.Series:
    values, timestamps = _read_timeseries_payload(payload, run_folder)
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values.ravel()[0])

    if timestamps.size:
        if values.ndim == 2 and values.shape[0] == timestamps.size:
            capacity_values = values[:, 0]
        elif values.ndim == 2 and values.shape[1] == timestamps.size:
            capacity_values = values[0, :]
        else:
            capacity_values = values.ravel()

        if capacity_values.size == timestamps.size:
            return pd.Series(capacity_values, index=_timestamps_to_index(timestamps))

    return float(values.ravel()[0])


def compute_market_step_technology_map(result_path: Path, csv_path: Path) -> pd.DataFrame:
    df_market_step_input = pd.read_csv(csv_path)

    fuelcost_types = df_market_step_input.loc[:, ["fuelcost", "type"]].dropna().copy()
    fuelcost_types["fuelcost_key"] = fuelcost_types["fuelcost"].round(6)
    fuelcost_to_types = (
        fuelcost_types.groupby("fuelcost_key")["type"]
        .unique()
        .apply(lambda x: sorted(x))
        .to_dict()
    )

    fuelcost_values = np.array(sorted(fuelcost_to_types.keys()), dtype=float)

    def lookup_types(price_val: float) -> List[str]:
        key = round(price_val, 6)
        types = fuelcost_to_types.get(key)
        if types:
            return list(types)
        if fuelcost_values.size:
            idx = np.abs(fuelcost_values - price_val).argmin()
            if abs(fuelcost_values[idx] - price_val) <= FUELCOST_TOL:
                return list(fuelcost_to_types[fuelcost_values[idx]])
        return []

    model = read_ltm_model(result_path)
    run_folder = result_path / "run_folder" / "emps"
    busbars_by_name = market_step_busbars_from_model(model)

    market_step_records = []
    for market_step in model.get("market_steps", []):
        market_step_name = str(market_step.get("name", ""))
        if not market_step_name:
            continue

        price_val = _timeseries_first_value(market_step.get("price", {}), run_folder)
        if price_val is None:
            logger.debug(f"Skipping market step without constant price: {market_step_name}")
            continue

        try:
            capacity = _timeseries_capacity(market_step.get("capacity", {}), run_folder)
        except (KeyError, OSError, ValueError, TypeError):
            logger.warning(f"Error processing market step capacity: {market_step_name}")
            capacity = 0.0

        # Force special market steps regardless of fuel cost mapping
        marker = f"{market_step_name} {market_step.get('#comment', '')}"
        fuel_type = market_step.get("fuel_type")
        if "FLOM" in marker:
            technology_types = ["spillage"]
        elif "RASJ" in marker:
            technology_types = ["rationing"]
        elif fuel_type:
            technology_types = [str(fuel_type)]
        else:
            technology_types = lookup_types(price_val)
            if not technology_types:
                technology_types = ["other"]

        market_step_records.append(
            {
                "market_step": market_step_name,
                "busbar": busbars_by_name.get(market_step_name),
                "price": price_val,
                "max_capacity": capacity,
                "technology_types": technology_types,
            }
        )

    return pd.DataFrame(market_step_records)


def build_merit_table(df_market_step_tech_map_area: pd.DataFrame) -> pd.DataFrame:
    df = df_market_step_tech_map_area.copy()
    df["technology_name"] = df["technology_types"].apply(normalize_technology_name)
    df = df[df["technology_name"].ne("unknown")]

    grouped = (
        df.groupby(["technology_name", "price"], sort=False)["max_capacity"]
        .apply(list)
        .reset_index(name="max_capacity")
    )
    grouped["max_capacity"] = grouped["max_capacity"].apply(sum_capacities)
    grouped = grouped.sort_values("price").reset_index(drop=True)
    duplicates = grouped["technology_name"].duplicated(keep=False)
    if duplicates.any():
        grouped.loc[duplicates, "technology_name"] = grouped.loc[duplicates].apply(
            lambda row: f"{row['technology_name']}@{row['price']:.6g}", axis=1
        )
    return grouped


def compute_dispatch_for_area(
    scenario: ScenarioResults,
    area: str,
    df_merit: pd.DataFrame,
    rationing_price: float,
    strict: bool = True,
) -> pd.DataFrame:
    df_sum_market_steps_matrix = scenario.get_market_steps_for_busbar(area)

    df_sum_market_steps = (
        df_sum_market_steps_matrix.stack()
        .to_frame("sum_market_steps")
        .swaplevel(0, 1)
        .sort_index()
        .rename_axis(["scenario", "timestamp"])
    )

    df_market_price_matrix = scenario.get_prices_for_busbar(area)
    df_market_price = (
        df_market_price_matrix.stack()
        .to_frame("market_price")
        .swaplevel(0, 1)
        .sort_index()
        .rename_axis(["scenario", "timestamp"])
    )
    try:
        fixed_nuclear_matrix = scenario.get_fixed_nuclear_for_busbar(area)
    except KeyError:
        fixed_nuclear_matrix = pd.DataFrame(
            0.0,
            index=df_market_price_matrix.index,
            columns=df_market_price_matrix.columns,
        )
    fixed_nuclear = (
        fixed_nuclear_matrix.stack()
        .to_frame("nuclear")
        .swaplevel(0, 1)
        .sort_index()
        .rename_axis(["scenario", "timestamp"])
    )

    df_sum_market_steps = df_sum_market_steps.reindex(df_market_price.index).fillna(0.0)
    # Spillage can be inferred as negative market-step production, drop from merit and add as separate column
    df_spillage = df_sum_market_steps["sum_market_steps"].clip(upper=0.0).copy(deep=True)
    df_merit = df_merit.drop(df_merit.index[df_merit["technology_name"] == "spillage"])

    df_dispatch = pd.DataFrame(
        0.0,
        index=df_market_price.index,
        columns=df_merit["technology_name"].tolist(),
    )

    # Fill capacities
    timestamps = df_market_price.index.get_level_values("timestamp").unique()

    for _, row in df_merit.iterrows():
        tech = row["technology_name"]
        cap = row["max_capacity"]
        if isinstance(cap, np.ndarray):
            if cap.size == 1:
                cap = float(cap.ravel()[0])
            else:
                if cap.size == len(timestamps):
                    cap = pd.Series(cap.ravel(), index=timestamps)
                else:
                    logger.warning(
                        f"Capacity array size mismatch for {tech} in {area}; using first value"
                    )
                    cap = float(cap.ravel()[0])
        if isinstance(cap, (int, float)):
            df_dispatch[tech] = cap
        elif isinstance(cap, pd.Series):
            if isinstance(cap.index, pd.DatetimeIndex):
                ts_tz = timestamps.tz
                if ts_tz is not None:
                    if cap.index.tz is None:
                        cap.index = cap.index.tz_localize(ts_tz)
                    elif cap.index.tz != ts_tz:
                        cap.index = cap.index.tz_convert(ts_tz)
                elif cap.index.tz is not None:
                    cap.index = cap.index.tz_convert(None)

            cap_aligned = cap.reindex(timestamps)
            ts_index = df_dispatch.index.get_level_values("timestamp")
            df_dispatch[tech] = ts_index.map(cap_aligned)
        else:
            df_dispatch[tech] = cap

    df_dispatch = df_dispatch.fillna(0.0)

    # Only keep techs that are in merit order at price (excluding rationing)
    merit_mask = ~df_merit["technology_name"].isin(["rationing"])
    for _, row in df_merit[merit_mask].iterrows():
        tech = row["technology_name"]
        df_dispatch[tech] = df_dispatch[tech].where(
            df_market_price["market_price"] >= row["price"], 0.0
        )

    # Dispatch to match sum_market_steps from cheapest to expensive
    remaining = df_sum_market_steps["sum_market_steps"].copy()
    for tech in df_merit.loc[merit_mask].sort_values("price")["technology_name"]:
        take = remaining.clip(upper=df_dispatch[tech], lower=0.0)
        df_dispatch[tech] = take
        remaining -= take

    # Rationing fills any remaining demand based on price triggers
    if "rationing" in df_dispatch.columns:
        cond_rat = df_market_price["market_price"] >= rationing_price
        df_dispatch["rationing"] = remaining.where(cond_rat, 0.0)
        remaining -= df_dispatch["rationing"]

    df_dispatch["spillage"] = df_spillage
    df_dispatch["total"] = df_dispatch.sum(axis=1)
    df_dispatch["sum_market_steps"] = df_sum_market_steps["sum_market_steps"]
    df_dispatch["market_price"] = df_market_price["market_price"]
    df_dispatch["diff"] = df_dispatch["total"] - df_dispatch["sum_market_steps"]

    fixed_nuclear = fixed_nuclear.reindex(df_dispatch.index).fillna(0.0)["nuclear"]
    if fixed_nuclear.abs().sum() > 0:
        if "nuclear" in df_dispatch.columns:
            df_dispatch["nuclear"] = df_dispatch["nuclear"] + fixed_nuclear
        else:
            df_dispatch["nuclear"] = fixed_nuclear

    if strict and df_dispatch["diff"].abs().mean() > 0.1:
        logger.warning(
            f"Significant difference between dispatched and market step sum for area {area}. "
            f"Mean diff: {df_dispatch['diff'].mean()}"
        )
        # raise ValueError("Dispatched generation does not match market step sum")

    return df_dispatch


def scenario_output_dir(scenario: ScenarioResults) -> Path:
    output_dir = processed_data_path_for_result(scenario.result_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dispatch(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    parquet_path = output_dir / f"{filename}.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


def should_include_area(area: str, prefixes: Iterable[str]) -> bool:
    return area.startswith(prefixes)


def process_scenario(scenario: ScenarioResults, csv_path: Path, strict: bool = True) -> pd.DataFrame:
    logger.info(
        f"Using processed time series and serialized market-step metadata for {scenario.name}."
    )
    df_market_step_tech_map = compute_market_step_technology_map(scenario.result_path, csv_path)

    rationing_rows = df_market_step_tech_map[
        df_market_step_tech_map["technology_types"].apply(lambda value: "rationing" in value)
    ]
    rationing_price = float(rationing_rows["price"].iloc[0]) if not rationing_rows.empty else float("inf")

    area_frames: List[pd.DataFrame] = []
    for area in sorted(scenario.get_busbar_names()):
        if not should_include_area(area, AREA_PREFIXES):
            continue
        df_area_map = df_market_step_tech_map[df_market_step_tech_map["busbar"] == area]
        if df_area_map.empty:
            logger.warning(f"No market steps for area {area} in {scenario.name}")
            continue

        merit = build_merit_table(df_area_map)
        dispatch = compute_dispatch_for_area(
            scenario,
            area,
            merit,
            rationing_price=rationing_price,
            strict=strict,
        )
        dispatch["area"] = area
        area_frames.append(dispatch)

    if not area_frames:
        raise ValueError(f"No dispatch frames computed for scenario {scenario.name}")

    df_all = pd.concat(area_frames)
    df_all = df_all.set_index("area", append=True).reorder_levels([2, 0, 1])
    df_all = df_all.sort_index().rename_axis(["area", "scenario", "timestamp"])
    return df_all


def _run_single(label: str, scenario: ScenarioResults) -> None:
    logger.info(f"Processing market dispatch for scenario {label} ({scenario.name})")
    df_all = process_scenario(scenario, GENERATOR_CSV, strict=True)
    output_dir = scenario_output_dir(scenario)
    output_path = save_dispatch(df_all, output_dir, "market_dispatch")
    logger.info(f"Saved dispatch to {output_path}")


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
            futures = {
                executor.submit(_run_single, label, scenario): label
                for label, scenario in scenarios.items()
            }
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
        for label, scenario in scenarios.items():
            _run_single(label, scenario)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process market dispatch results.")
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
