from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from scripts.processed_results import ProcessedScenarioResults


MARKET_STEP_TECHNOLOGY_COLUMNS = ("biomass", "fossil_gas", "fossil_other", "rationing", "market_spillage")


def _zero_like(base: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=base.index, columns=base.columns)


def _metric_or_zero(
    scenario: ProcessedScenarioResults,
    area: str,
    metric: str,
    base: pd.DataFrame,
) -> pd.DataFrame:
    try:
        return scenario.get_busbar_metric(area, metric)
    except KeyError:
        return _zero_like(base)


def _load_or_none(scenario: ProcessedScenarioResults, area: str, metric: str) -> pd.DataFrame | None:
    try:
        return scenario.get_busbar_metric(area, metric)
    except KeyError:
        return None


def _to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    long_df = df.stack().rename(value_name).reset_index()
    long_df.columns = ["timestamp", "scenario", value_name]
    return long_df


def _merge_long_frames(frames: dict[str, pd.DataFrame], area: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name, frame in frames.items():
        long_frame = _to_long(frame, name)
        if merged is None:
            merged = long_frame
        else:
            merged = merged.merge(long_frame, on=["timestamp", "scenario"], how="left")

    if merged is None:
        return pd.DataFrame()
    merged["area"] = area
    return merged


def load_processed_dispatch_data(
    scenario_path: Path,
    areas: Iterable[str] | None = None,
    require_market_step_technologies: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    scenario = ProcessedScenarioResults.from_result_path(scenario_path)
    if scenario is None:
        return None

    if require_market_step_technologies:
        columns = set(pq.ParquetFile(scenario.data_path).schema_arrow.names)
        missing = [column for column in MARKET_STEP_TECHNOLOGY_COLUMNS if column not in columns]
        if missing:
            raise RuntimeError(
                f"{scenario.data_path} is missing processed market-step technology columns "
                f"{missing}. Use the Zenodo processed-data archive that matches this script version."
            )

    selected_areas = list(areas) if areas is not None else scenario.get_busbar_names()
    export_rows: list[pd.DataFrame] = []
    process_rows: list[pd.DataFrame] = []

    for area in selected_areas:
        base = _load_or_none(scenario, area, "load")
        if base is None:
            base = _load_or_none(scenario, area, "price")
        if base is None:
            continue

        fixed_nuclear = _metric_or_zero(scenario, area, "fixed_nuclear", base)
        try:
            total_nuclear = scenario.get_total_nuclear_for_busbar(area)
        except KeyError:
            total_nuclear = fixed_nuclear

        market_steps = _metric_or_zero(scenario, area, "market_steps", base)
        flexible_nuclear = total_nuclear.sub(fixed_nuclear, fill_value=0.0).clip(lower=0.0)
        market_steps_without_flexible_nuclear = market_steps.sub(flexible_nuclear, fill_value=0.0)

        export_rows.append(
            _merge_long_frames(
                {
                    "load": _metric_or_zero(scenario, area, "load", base),
                    "hydro": _metric_or_zero(scenario, area, "hydro", base),
                    "onshore_wind": _metric_or_zero(scenario, area, "onshore_wind", base),
                    "offshore_wind": _metric_or_zero(scenario, area, "offshore_wind", base),
                    "nuclear": total_nuclear.reindex(index=base.index, columns=base.columns, fill_value=0.0),
                    "solar": _metric_or_zero(scenario, area, "solar", base),
                    "market_steps": market_steps_without_flexible_nuclear,
                },
                area,
            )
        )

        process_rows.append(
            _merge_long_frames(
                {
                    "market_price": _metric_or_zero(scenario, area, "price", base),
                    "nuclear": total_nuclear.reindex(index=base.index, columns=base.columns, fill_value=0.0),
                    "biomass": _metric_or_zero(scenario, area, "biomass", base),
                    "fossil_gas": _metric_or_zero(scenario, area, "fossil_gas", base),
                    "fossil_other": _metric_or_zero(scenario, area, "fossil_other", base),
                    "spillage": _metric_or_zero(scenario, area, "market_spillage", base),
                    "rationing": _metric_or_zero(scenario, area, "rationing", base),
                },
                area,
            )
        )

    export_df = pd.concat(export_rows, ignore_index=True) if export_rows else pd.DataFrame()
    process_df = pd.concat(process_rows, ignore_index=True) if process_rows else pd.DataFrame()

    if export_df.empty or process_df.empty:
        return None

    return (
        export_df.set_index(["area", "scenario", "timestamp"]).sort_index(),
        process_df.set_index(["area", "scenario", "timestamp"]).sort_index(),
    )
