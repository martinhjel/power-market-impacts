from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import logger
from scripts.processed_results import (
    DEFAULT_OUTPUT_ROOT,
    add_project_paths,
    df_from_ltm_result,
    make_timeseries_records,
    normalise_processed_frame,
    processed_data_path,
    write_metadata,
)
from scripts.merit_order import (
    MarketStepRecord,
    capacity_to_base_axes as _capacity_to_result_axes,
    market_step_fuel_type,
    market_step_price,
    object_busbar_name as _object_busbar_name,
    object_name as _object_name,
    reconstruct_named_market_step_dispatch,
)

add_project_paths(PROJECT_ROOT)

try:
    from uprate_hydro import uprate_values as UPRATED_PLANTS  # type: ignore
except Exception:
    UPRATED_PLANTS = {}

from nuclear_modeling import (
    HISTORIC_NUCLEAR_FLEXIBLE_PREFIX,
    HISTORIC_NUCLEAR_PREFIX,
    NEW_NUCLEAR_FIRM_PREFIX,
    NEW_NUCLEAR_FLEXIBLE_PREFIX,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frequently used LTM result time series into "
            "ltm_processed/<model>/<scenario>/processed_data.parquet."
        )
    )
    parser.add_argument("--model-folder", required=True, help="Folder under ltm_output")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Scenario folder names. Defaults to every scenario folder with run_folder/emps.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--areas", nargs="+", help="Only process selected busbars/areas.")
    parser.add_argument("--no-dclines", action="store_true", help="Skip DC line flows.")
    parser.add_argument(
        "--reservoir-mode",
        choices=["none", "uprated", "all"],
        default="uprated",
        help=(
            "Reservoir records to store. 'uprated' stores only reservoirs listed in "
            "data/uprate_hydro.py; 'all' can be much larger."
        ),
    )
    parser.add_argument(
        "--no-reservoir-aggregates",
        action="store_true",
        help="Skip area-level aggregate spill/discharge derived from individual reservoirs.",
    )
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of scenario worker processes. Values above 1 enable multiprocessing. "
            "Defaults to serial unless --parallel is used."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process scenarios in parallel using up to four workers unless --workers is set.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class ProcessingOptions:
    model_folder: str
    output_root: str
    overwrite: bool
    areas: tuple[str, ...] | None
    no_dclines: bool
    reservoir_mode: str
    no_reservoir_aggregates: bool
    compression: str


@dataclass
class RenewableObjectIndex:
    solars_by_area: dict[str, list[object]]
    winds_by_area: dict[str, list[object]]


@dataclass
class LTMObjectMetadata:
    names: dict[str, str]
    busbars: dict[str, str]
    fuel_types: dict[str, str]

    def name_for(self, obj) -> str:
        name = _object_name(obj)
        return self.names.get(name, name)

    def busbar_for(self, obj) -> str | None:
        name = _object_name(obj)
        return self.busbars.get(name) or _object_busbar_name(obj)

    def fuel_type_for(self, obj) -> str:
        name = _object_name(obj)
        return self.fuel_types.get(name) or market_step_fuel_type(obj)


@contextmanager
def log_duration(label: str):
    start = perf_counter()
    logger.info("Starting %s", label)
    try:
        yield
    finally:
        logger.info("Finished %s in %.1f s", label, perf_counter() - start)


def options_from_args(args: argparse.Namespace) -> ProcessingOptions:
    return ProcessingOptions(
        model_folder=args.model_folder,
        output_root=args.output_root,
        overwrite=args.overwrite,
        areas=tuple(args.areas) if args.areas else None,
        no_dclines=args.no_dclines,
        reservoir_mode=args.reservoir_mode,
        no_reservoir_aggregates=args.no_reservoir_aggregates,
        compression=args.compression,
    )


def resolve_worker_count(args: argparse.Namespace, scenario_count: int) -> int:
    if scenario_count <= 1:
        return 1
    if args.workers is not None:
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        return min(args.workers, scenario_count)
    if args.parallel:
        return min(4, os.cpu_count() or 1, scenario_count)
    return 1


class IncrementalParquetWriter:
    def __init__(self, output_path: Path, compression: str = "zstd"):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression = None if compression == "none" else compression
        self.writer: pq.ParquetWriter | None = None
        self.row_counts: dict[str, int] = {}
        self.write_calls = 0
        self.write_seconds = 0.0

    def write(self, df: pd.DataFrame, *, normalised: bool = False) -> None:
        if df.empty:
            return
        start = perf_counter()
        try:
            if not normalised:
                df = normalise_processed_frame(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self.writer is None:
                self.writer = pq.ParquetWriter(self.output_path, table.schema, compression=self.compression)
            else:
                table = table.cast(self.writer.schema)
            self.writer.write_table(table)
            self.write_calls += 1
            for record_type, count in df["record_type"].value_counts().items():
                self.row_counts[str(record_type)] = self.row_counts.get(str(record_type), 0) + int(count)
        finally:
            self.write_seconds += perf_counter() - start

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _result_path(model_folder: str, scenario_name: str) -> Path:
    return PROJECT_ROOT / "ltm_output" / model_folder / scenario_name


def _normalise_ltm_comment_name(comment: str) -> str:
    name = comment.split(" -> ", 1)[0].strip()
    for prefix in ("market_step_", "wind_", "solar_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def read_ltm_object_metadata(result_path: Path) -> LTMObjectMetadata:
    model_json = result_path / "run_folder" / "emps" / "ltm_model.json"
    if not model_json.exists():
        return LTMObjectMetadata(names={}, busbars={}, fuel_types={})

    with open(model_json) as f:
        data = json.load(f).get("model", {})

    names: dict[str, str] = {}
    fuel_types: dict[str, str] = {}
    for collection_name in ("solar", "wind", "market_steps"):
        for item in data.get(collection_name, []):
            ltm_name = item.get("name")
            comment = item.get("#comment")
            if ltm_name and comment:
                names[str(ltm_name)] = _normalise_ltm_comment_name(str(comment))
            if collection_name == "market_steps" and ltm_name and item.get("fuel_type"):
                fuel_types[str(ltm_name)] = str(item["fuel_type"])

    busbars: dict[str, str] = {}
    for connection in data.get("connections", []):
        from_name = connection.get("from")
        to_name = connection.get("to")
        comment = str(connection.get("#comment", ""))
        if not from_name or not to_name or "busbar_" not in comment:
            continue
        busbars[str(from_name)] = str(to_name)

    return LTMObjectMetadata(names=names, busbars=busbars, fuel_types=fuel_types)


def validate_raw_result_complete(result_path: Path) -> None:
    expected_files = [
        result_path / "results" / "results.h5",
        result_path / "run_folder" / "emps" / "emps_sim.out",
    ]
    missing = [path.relative_to(result_path) for path in expected_files if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            f"Incomplete LTM result for {result_path.name}: missing {missing_text}. "
            "The scenario run likely failed before simulation completed."
        )


def discover_scenarios(model_folder: str) -> list[str]:
    root = PROJECT_ROOT / "ltm_output" / model_folder
    if not root.exists():
        raise FileNotFoundError(f"Model folder not found: {root}")
    scenarios = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "run_folder" / "emps").exists():
            scenarios.append(path.name)
    return scenarios


def read_run_config(result_path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    for path in (result_path.parent / "config.txt", result_path / "config.txt"):
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()
    return config


def _safe_df(label: str, func) -> pd.DataFrame | None:
    try:
        result = func() if callable(func) else func
        return df_from_ltm_result(result)
    except Exception as exc:
        logger.debug("Skipping unavailable %s: %s", label, exc)
        return None


def _safe_reservoir_df(label: str, func) -> pd.DataFrame | None:
    try:
        return df_from_ltm_result(func(time_axis=True))
    except TypeError:
        try:
            return df_from_ltm_result(func())
        except Exception as exc:
            logger.debug("Skipping unavailable %s: %s", label, exc)
            return None
    except Exception as exc:
        logger.debug("Skipping unavailable %s: %s", label, exc)
        return None


def _index_by_busbar(objects: Iterable[object], metadata: LTMObjectMetadata) -> dict[str, list[object]]:
    indexed: dict[str, list[object]] = {}
    for obj in objects:
        area = metadata.busbar_for(obj)
        if not area:
            continue
        indexed.setdefault(area, []).append(obj)
    return indexed


def build_renewable_index(model, metadata: LTMObjectMetadata) -> RenewableObjectIndex:
    try:
        solars = model.solar()
    except Exception:
        solars = []
    try:
        winds = model.wind()
    except Exception:
        winds = []
    return RenewableObjectIndex(
        solars_by_area=_index_by_busbar(solars, metadata),
        winds_by_area=_index_by_busbar(winds, metadata),
    )


def build_market_step_index(model, metadata: LTMObjectMetadata) -> dict[str, list[object]]:
    try:
        market_steps = model.market_steps()
    except Exception:
        return {}
    return _index_by_busbar(market_steps, metadata)


def market_step_records_for_area(
    market_steps_by_area: dict[str, list[object]],
    metadata: LTMObjectMetadata,
    area: str,
    base: pd.DataFrame | None,
) -> list[MarketStepRecord]:
    if base is None:
        return []
    return [
        MarketStepRecord(
            name=metadata.name_for(market_step),
            busbar=area,
            price=market_step_price(market_step),
            capacity=_capacity_to_result_axes(market_step, base),
            fuel_type=metadata.fuel_type_for(market_step),
        )
        for market_step in market_steps_by_area.get(area, [])
    ]


def renewable_generation_for_area(
    renewable_index: RenewableObjectIndex,
    metadata: LTMObjectMetadata,
    area: str,
    base: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    if base is None:
        return {}

    totals = {
        "solar": pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32"),
        "onshore_wind": pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32"),
        "offshore_wind": pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32"),
        "fixed_nuclear": pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32"),
        "historic_nuclear": pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32"),
        "historic_nuclear_available": pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32"),
        "_new_nuclear_firm": pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32"),
        "_new_nuclear_firm_available": pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32"),
    }

    for solar in renewable_index.solars_by_area.get(area, []):
        totals["solar"] = totals["solar"].add(_capacity_to_result_axes(solar, base), fill_value=0.0)

    for wind in renewable_index.winds_by_area.get(area, []):
        name = metadata.name_for(wind).lower()
        if "nuclear" in name:
            capacity = _capacity_to_result_axes(wind, base)
            totals["fixed_nuclear"] = totals["fixed_nuclear"].add(capacity, fill_value=0.0)
            if name.startswith(HISTORIC_NUCLEAR_PREFIX):
                totals["historic_nuclear"] = totals["historic_nuclear"].add(capacity, fill_value=0.0)
                totals["historic_nuclear_available"] = totals["historic_nuclear_available"].add(
                    capacity, fill_value=0.0
                )
            elif name.startswith(NEW_NUCLEAR_FIRM_PREFIX):
                totals["_new_nuclear_firm"] = totals["_new_nuclear_firm"].add(capacity, fill_value=0.0)
                totals["_new_nuclear_firm_available"] = totals["_new_nuclear_firm_available"].add(
                    capacity, fill_value=0.0
                )
            continue
        elif "_off" in name or "offshore" in name:
            key = "offshore_wind"
        elif "_on" in name or "onshore" in name:
            key = "onshore_wind"
        else:
            key = "onshore_wind"
        totals[key] = totals[key].add(_capacity_to_result_axes(wind, base), fill_value=0.0)

    return {name: df.astype("float32") for name, df in totals.items()}


def reconstruct_flexible_nuclear_from_records(
    *,
    records: list[MarketStepRecord],
    base: pd.DataFrame | None,
    market_price: pd.DataFrame | None,
    total_market_steps: pd.DataFrame | None,
    is_target,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if base is None:
        return None, None

    target_records = [record for record in records if is_target(record)]
    if not target_records:
        zero = _zero_like(base)
        return zero, zero

    available = pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32")
    for record in target_records:
        available = available.add(record.capacity, fill_value=0.0)

    if market_price is None or total_market_steps is None:
        return _zero_like(base), available.astype("float32")

    generated, available = reconstruct_named_market_step_dispatch(
        market_steps=records,
        market_price=market_price,
        total_market_steps=total_market_steps,
        is_target=is_target,
    )
    return generated, available


MARKET_STEP_TECHNOLOGY_COLUMNS = ("biomass", "fossil_gas", "fossil_other", "rationing")


def _market_step_technology(record: MarketStepRecord) -> str | None:
    name = record.name.lower()
    if "rasj" in name:
        return "rationing"

    fuel_type = record.fuel_type.lower()
    if fuel_type in MARKET_STEP_TECHNOLOGY_COLUMNS:
        return fuel_type

    for technology in ("biomass", "fossil_gas", "fossil_other"):
        if technology in name:
            return technology
    return None


def reconstruct_market_step_technologies_from_records(
    *,
    records: list[MarketStepRecord],
    base: pd.DataFrame | None,
    market_price: pd.DataFrame | None,
    total_market_steps: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    if base is None:
        return {}

    frames = {name: _zero_like(base) for name in MARKET_STEP_TECHNOLOGY_COLUMNS}
    frames["market_spillage"] = _zero_like(base)

    if total_market_steps is not None:
        frames["market_spillage"] = total_market_steps.reindex(
            index=base.index,
            columns=base.columns,
            fill_value=0.0,
        ).clip(upper=0.0).astype("float32")

    if market_price is None or total_market_steps is None:
        return frames

    remaining = total_market_steps.reindex(index=base.index, columns=base.columns, fill_value=0.0).astype("float32")
    for record in sorted(records, key=lambda item: (item.price, item.name)):
        capacity = record.capacity.reindex(index=base.index, columns=base.columns, fill_value=0.0).astype("float32")
        eligible_capacity = capacity.where(market_price >= record.price, 0.0)
        take = remaining.clip(lower=0.0)
        take = take.where(take <= eligible_capacity, eligible_capacity)

        technology = _market_step_technology(record)
        if technology in frames:
            frames[technology] = frames[technology].add(take, fill_value=0.0).astype("float32")

        remaining = remaining - take

    return frames


def _is_generic_nuclear_market_step(record: MarketStepRecord) -> bool:
    name = record.name.lower()
    explicit_prefixes = (
        HISTORIC_NUCLEAR_FLEXIBLE_PREFIX,
        NEW_NUCLEAR_FLEXIBLE_PREFIX,
        HISTORIC_NUCLEAR_PREFIX,
        NEW_NUCLEAR_FIRM_PREFIX,
    )
    return (
        not name.startswith(explicit_prefixes)
        and (record.fuel_type.lower() == "nuclear" or "nuclear" in name)
    )


def is_historic_flexible_nuclear(record: MarketStepRecord) -> bool:
    name = record.name.lower()
    if name.startswith(HISTORIC_NUCLEAR_FLEXIBLE_PREFIX):
        return True
    return _is_generic_nuclear_market_step(record) and not record.busbar.startswith("NO")


def is_new_flexible_nuclear(record: MarketStepRecord) -> bool:
    name = record.name.lower()
    if name.startswith(NEW_NUCLEAR_FLEXIBLE_PREFIX):
        return True
    return _is_generic_nuclear_market_step(record) and record.busbar.startswith("NO")


def _add_frames(a: pd.DataFrame | None, b: pd.DataFrame | None) -> pd.DataFrame | None:
    if b is None:
        return a
    if a is None:
        return b
    return a.add(b, fill_value=0.0).astype("float32")


def _zero_like(base: pd.DataFrame | None) -> pd.DataFrame | None:
    if base is None:
        return None
    return pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32")


def aggregate_reservoir_metrics(busbar) -> dict[str, pd.DataFrame]:
    totals: dict[str, pd.DataFrame | None] = {
        "reservoir_spill": None,
        "reservoir_discharge": None,
    }
    for reservoir in busbar.reservoirs():
        spill = _safe_reservoir_df(f"{reservoir.name}.spill", reservoir.spill)
        discharge = _safe_reservoir_df(f"{reservoir.name}.discharge", reservoir.discharge)
        totals["reservoir_spill"] = _add_frames(totals["reservoir_spill"], spill)
        totals["reservoir_discharge"] = _add_frames(totals["reservoir_discharge"], discharge)
    return {name: df for name, df in totals.items() if df is not None}


def _target_reservoirs() -> dict[str, dict[str, str]]:
    targets = {}
    for plant_name, info in UPRATED_PLANTS.items():
        reservoirs = info.get("reservoirs", [])
        if not reservoirs:
            continue
        reservoir_name = reservoirs[0]
        targets[plant_name] = {
            "area": info.get("elspot_area", ""),
            "reservoir_name": reservoir_name,
            "reservoir_ltm_name": f"reservoir_{reservoir_name.lower()}",
        }
    return targets


def _normalised_reservoir_name(name: str) -> str:
    import re

    name = name.lower()
    for prefix in ("reservoir_", "res_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return re.sub(r"_[0-9a-f]{4}$", "", name)


def find_reservoir(target_name: str, reservoirs: Iterable[object]):
    by_name = {r.name.lower(): r for r in reservoirs}
    names = [
        target_name.lower(),
        f"res_{target_name.lower()}",
        f"reservoir_{target_name.lower()}",
    ]
    for name in names:
        if name in by_name:
            return by_name[name]

    target = _normalised_reservoir_name(target_name)
    if len(target) < 5:
        return None
    candidates = [
        reservoir
        for name, reservoir in by_name.items()
        if _normalised_reservoir_name(name).startswith(target[:5])
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def process_busbars(
    *,
    writer: IncrementalParquetWriter,
    model,
    areas: set[str] | None,
    metadata: LTMObjectMetadata,
) -> list[str]:
    busbars = {busbar.name: busbar for busbar in model.busbars()}
    selected = [area for area in sorted(busbars) if areas is None or area in areas]
    renewable_index = build_renewable_index(model, metadata)
    market_steps_by_area = build_market_step_index(model, metadata)
    logger.info("Processing %d busbars", len(selected))

    for area in selected:
        busbar = busbars[area]
        logger.info("  busbar=%s", area)
        frames = {
            "price": _safe_df(f"{area}.price", busbar.market_result_price),
            "load": _safe_df(f"{area}.load", busbar.sum_load),
            "hydro": _safe_df(f"{area}.hydro", busbar.sum_hydro_production),
            "reservoir": _safe_df(f"{area}.reservoir", busbar.sum_reservoir),
            "market_steps": _safe_df(f"{area}.market_steps", busbar.sum_production_from_market_steps),
        }
        base = next((df for df in frames.values() if df is not None and len(df.index) > 100), None)
        renewable_frames = renewable_generation_for_area(renewable_index, metadata, area, base)
        market_step_records = market_step_records_for_area(market_steps_by_area, metadata, area, base)
        new_firm = renewable_frames.pop("_new_nuclear_firm", _zero_like(base))
        new_firm_available = renewable_frames.pop("_new_nuclear_firm_available", _zero_like(base))
        historic_flexible, historic_flexible_available = reconstruct_flexible_nuclear_from_records(
            records=market_step_records,
            base=base,
            market_price=frames.get("price"),
            total_market_steps=frames.get("market_steps"),
            is_target=is_historic_flexible_nuclear,
        )
        new_flexible, new_flexible_available = reconstruct_flexible_nuclear_from_records(
            records=market_step_records,
            base=base,
            market_price=frames.get("price"),
            total_market_steps=frames.get("market_steps"),
            is_target=is_new_flexible_nuclear,
        )
        market_step_technology_frames = reconstruct_market_step_technologies_from_records(
            records=market_step_records,
            base=base,
            market_price=frames.get("price"),
            total_market_steps=frames.get("market_steps"),
        )
        frames.update(renewable_frames)
        frames.update(market_step_technology_frames)
        if base is not None:
            historic_firm = frames.get("historic_nuclear", _zero_like(base))
            historic_firm_available = frames.get("historic_nuclear_available", _zero_like(base))
            historic_flexible = historic_flexible if historic_flexible is not None else _zero_like(base)
            historic_flexible_available = (
                historic_flexible_available if historic_flexible_available is not None else _zero_like(base)
            )
            new_firm = new_firm if new_firm is not None else _zero_like(base)
            new_firm_available = new_firm_available if new_firm_available is not None else _zero_like(base)
            new_flexible = new_flexible if new_flexible is not None else _zero_like(base)
            new_flexible_available = (
                new_flexible_available if new_flexible_available is not None else _zero_like(base)
            )
            frames["historic_nuclear"] = _add_frames(historic_firm, historic_flexible)
            frames["historic_nuclear_available"] = _add_frames(historic_firm_available, historic_flexible_available)
            frames["new_nuclear"] = _add_frames(new_firm, new_flexible)
            frames["new_nuclear_available"] = _add_frames(new_firm_available, new_flexible_available)
            frames["total_nuclear"] = _add_frames(frames["historic_nuclear"], frames["new_nuclear"])
            frames["total_nuclear_available"] = _add_frames(
                frames["historic_nuclear_available"], frames["new_nuclear_available"]
            )
        frames = {k: v for k, v in frames.items() if v is not None}
        writer.write(
            make_timeseries_records(
                record_type="busbar",
                area=area,
                entity=area,
                frames=frames,
            ),
            normalised=True,
        )

    return selected


def process_reservoir_aggregates(*, writer: IncrementalParquetWriter, model, areas: set[str] | None) -> None:
    busbars = {busbar.name: busbar for busbar in model.busbars()}
    selected = [area for area in sorted(busbars) if areas is None or area in areas]
    logger.info("Processing reservoir aggregate spill/discharge for %d busbars", len(selected))
    for area in selected:
        busbar = busbars[area]
        frames = aggregate_reservoir_metrics(busbar)
        if not frames:
            continue
        writer.write(
            make_timeseries_records(
                record_type="busbar",
                area=area,
                entity=area,
                frames=frames,
            ),
            normalised=True,
        )


def process_reservoir_records(
    *,
    writer: IncrementalParquetWriter,
    model,
    areas: set[str] | None,
    reservoir_mode: str,
) -> None:
    if reservoir_mode == "none":
        return

    busbars = {busbar.name: busbar for busbar in model.busbars()}
    if reservoir_mode == "uprated":
        targets = _target_reservoirs()
        logger.info("Processing %d uprated reservoir records", len(targets))
        for plant_name, target in targets.items():
            area = target["area"]
            if areas is not None and area not in areas:
                continue
            if area not in busbars:
                continue
            reservoir = find_reservoir(target["reservoir_name"], busbars[area].reservoirs())
            if reservoir is None:
                logger.warning("  missing reservoir target %s in %s", plant_name, area)
                continue
            write_reservoir_record(writer, area, plant_name, reservoir)
        return

    logger.info("Processing all reservoir records")
    for area, busbar in sorted(busbars.items()):
        if areas is not None and area not in areas:
            continue
        for reservoir in busbar.reservoirs():
            write_reservoir_record(writer, area, reservoir.name, reservoir)


def read_reservoir_record_frames(reservoir) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame | None] = {
        "reservoir_production": _safe_reservoir_df(f"{reservoir.name}.production", reservoir.production),
        "reservoir_level": _safe_reservoir_df(f"{reservoir.name}.reservoir", reservoir.reservoir),
        "reservoir_spill": _safe_reservoir_df(f"{reservoir.name}.spill", reservoir.spill),
        "reservoir_discharge": _safe_reservoir_df(f"{reservoir.name}.discharge", reservoir.discharge),
    }
    return {k: v for k, v in frames.items() if v is not None}


def write_reservoir_record(
    writer: IncrementalParquetWriter,
    area: str,
    entity: str,
    reservoir,
    frames: dict[str, pd.DataFrame] | None = None,
) -> None:
    if frames is None:
        frames = read_reservoir_record_frames(reservoir)
    writer.write(
        make_timeseries_records(
            record_type="reservoir",
            area=area,
            entity=entity,
            object_name=reservoir.name,
            frames=frames,
        ),
        normalised=True,
    )


def process_all_reservoir_records_and_aggregates(
    *,
    writer: IncrementalParquetWriter,
    model,
    areas: set[str] | None,
) -> None:
    busbars = {busbar.name: busbar for busbar in model.busbars()}
    selected = [(area, busbars[area]) for area in sorted(busbars) if areas is None or area in areas]
    logger.info("Processing all reservoir records and aggregate spill/discharge for %d busbars", len(selected))

    for area, busbar in selected:
        totals: dict[str, pd.DataFrame | None] = {
            "reservoir_spill": None,
            "reservoir_discharge": None,
        }
        for reservoir in busbar.reservoirs():
            frames = read_reservoir_record_frames(reservoir)
            totals["reservoir_spill"] = _add_frames(totals["reservoir_spill"], frames.get("reservoir_spill"))
            totals["reservoir_discharge"] = _add_frames(
                totals["reservoir_discharge"], frames.get("reservoir_discharge")
            )
            write_reservoir_record(writer, area, reservoir.name, reservoir, frames=frames)

        aggregate_frames = {name: df for name, df in totals.items() if df is not None}
        if not aggregate_frames:
            continue
        writer.write(
            make_timeseries_records(
                record_type="busbar",
                area=area,
                entity=area,
                frames=aggregate_frames,
            ),
            normalised=True,
        )


def process_dclines(*, writer: IncrementalParquetWriter, model) -> None:
    try:
        dclines = model.dclines()
    except Exception:
        return
    logger.info("Processing %d DC lines", len(dclines))
    for dcline in dclines:
        flow_result = getattr(dcline, "transmission_results", None)
        if flow_result is None:
            logger.warning("Skipping DC line %s: no transmission_results on raw LTM object", dcline.name)
            continue
        df = _safe_df(f"{dcline.name}.flow", flow_result)
        if df is None:
            continue
        writer.write(
            make_timeseries_records(
                record_type="dcline",
                area="",
                entity=dcline.name,
                frames={"flow": df},
            ),
            normalised=True,
        )


def process_scenario(options: ProcessingOptions, scenario_name: str) -> Path:
    from lpr_sintef_bifrost.ltm import LTM

    result_path = _result_path(options.model_folder, scenario_name)
    output_path = processed_data_path(PROJECT_ROOT, options.model_folder, scenario_name, options.output_root)
    metadata_path = output_path.with_name("metadata.json")

    if output_path.exists() and not options.overwrite:
        logger.info("Skipping existing processed data: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    validate_raw_result_complete(result_path)
    object_metadata = read_ltm_object_metadata(result_path)

    with log_duration(f"{scenario_name}: load LTM result"):
        logger.info("Loading LTM result: %s", result_path)
        session = LTM.session_from_folder(result_path / "run_folder" / "emps")
        model = session.model
    areas = set(options.areas) if options.areas else None
    run_config = read_run_config(result_path)

    writer = IncrementalParquetWriter(output_path, compression=options.compression)
    try:
        with log_duration(f"{scenario_name}: busbars"):
            process_busbars(writer=writer, model=model, areas=areas, metadata=object_metadata)

        if options.reservoir_mode == "all" and not options.no_reservoir_aggregates:
            with log_duration(f"{scenario_name}: all reservoir records + aggregates"):
                process_all_reservoir_records_and_aggregates(writer=writer, model=model, areas=areas)
        else:
            if not options.no_reservoir_aggregates:
                with log_duration(f"{scenario_name}: reservoir aggregates"):
                    process_reservoir_aggregates(writer=writer, model=model, areas=areas)
            with log_duration(f"{scenario_name}: reservoir records ({options.reservoir_mode})"):
                process_reservoir_records(
                    writer=writer,
                    model=model,
                    areas=areas,
                    reservoir_mode=options.reservoir_mode,
                )

        if not options.no_dclines:
            with log_duration(f"{scenario_name}: dclines"):
                process_dclines(writer=writer, model=model)
    finally:
        writer.close()

    write_metadata(
        metadata_path=metadata_path,
        model_folder=options.model_folder,
        scenario_name=scenario_name,
        source_result_path=result_path,
        processed_data_path=output_path,
        options={
            "areas": sorted(areas) if areas else None,
            "dclines": not options.no_dclines,
            "reservoir_mode": options.reservoir_mode,
            "reservoir_aggregates": not options.no_reservoir_aggregates,
            "compression": options.compression,
            "run_config": run_config,
        },
        row_counts=writer.row_counts,
    )
    logger.info("Wrote %s", output_path)
    logger.info("Row counts: %s", writer.row_counts)
    logger.info("Parquet writes: %d calls, %.1f s", writer.write_calls, writer.write_seconds)
    return output_path


def process_scenarios_parallel(
    options: ProcessingOptions,
    scenarios: list[str],
    worker_count: int,
) -> None:
    failures: list[tuple[str, str]] = []
    logger.info("Processing scenarios with %d worker processes", worker_count)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(process_scenario, options, scenario_name): scenario_name
            for scenario_name in scenarios
        }
        for future in as_completed(futures):
            scenario_name = futures[future]
            try:
                output_path = future.result()
                logger.info("Finished %s -> %s", scenario_name, output_path)
            except Exception as exc:
                failures.append((scenario_name, str(exc)))
                logger.exception("Failed processing %s", scenario_name)

    if failures:
        summary = "; ".join(f"{scenario}: {error}" for scenario, error in failures)
        raise SystemExit(f"{len(failures)} scenario(s) failed: {summary}")


def main() -> None:
    args = parse_args()
    scenarios = args.scenarios or discover_scenarios(args.model_folder)
    if not scenarios:
        raise SystemExit(f"No scenarios found for model folder {args.model_folder}")

    options = options_from_args(args)
    worker_count = resolve_worker_count(args, len(scenarios))
    logger.info("Processing %d scenarios from %s", len(scenarios), args.model_folder)
    if worker_count == 1:
        for scenario_name in scenarios:
            process_scenario(options, scenario_name)
        return

    process_scenarios_parallel(options, scenarios, worker_count)


if __name__ == "__main__":
    main()
