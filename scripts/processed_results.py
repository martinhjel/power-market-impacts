from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCHEMA_VERSION = 3
DEFAULT_OUTPUT_ROOT = "ltm_processed"

STRING_COLUMNS = ["record_type", "area", "entity", "object_name"]
NUMERIC_COLUMNS = [
    "price",
    "load",
    "hydro",
    "reservoir",
    "market_steps",
    "solar",
    "onshore_wind",
    "offshore_wind",
    "fixed_nuclear",
    "historic_nuclear",
    "historic_nuclear_available",
    "new_nuclear",
    "new_nuclear_available",
    "total_nuclear",
    "total_nuclear_available",
    "biomass",
    "fossil_gas",
    "fossil_other",
    "rationing",
    "market_spillage",
    "reservoir_production",
    "reservoir_level",
    "reservoir_spill",
    "reservoir_discharge",
    "flow",
]
BASE_COLUMNS = ["schema_version", *STRING_COLUMNS, "timestamp", "weather_year"]
PROCESSED_COLUMNS = [*BASE_COLUMNS, *NUMERIC_COLUMNS]


def df_from_ltm_result(result) -> pd.DataFrame:
    try:
        from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result
    except ImportError:
        from lpr_sintef_bifrost.utils.dataframe import df_from_numpy_array_reference as df_from_pyltm_result

    return df_from_pyltm_result(result)


def project_root_from_result_path(result_path: Path) -> Path:
    result_path = Path(result_path).absolute()
    parts = result_path.parts
    if "ltm_output" not in parts:
        return Path.cwd()
    idx = parts.index("ltm_output")
    if idx == 0:
        return Path("/")
    return Path(*parts[:idx])


def infer_model_and_scenario(result_path: Path) -> tuple[str, str]:
    result_path = Path(result_path).absolute()
    parts = result_path.parts
    if "ltm_output" not in parts:
        raise ValueError(f"Cannot infer model/scenario from path outside ltm_output: {result_path}")
    idx = parts.index("ltm_output")
    try:
        return parts[idx + 1], parts[idx + 2]
    except IndexError as exc:
        raise ValueError(f"Expected ltm_output/<model>/<scenario>, got: {result_path}") from exc


def processed_data_path_for_result(
    result_path: Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    project_root = project_root_from_result_path(result_path)
    model_folder, scenario_name = infer_model_and_scenario(result_path)
    return project_root / output_root / model_folder / scenario_name / "processed_data.parquet"


def processed_metadata_path_for_result(
    result_path: Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    return processed_data_path_for_result(result_path, output_root).with_name("metadata.json")


def processed_data_path(
    project_root: Path,
    model_folder: str,
    scenario_name: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    return Path(project_root) / output_root / model_folder / scenario_name / "processed_data.parquet"


def _normalise_weather_years(columns: Iterable[object]) -> list[int]:
    years = []
    for i, column in enumerate(columns):
        try:
            years.append(int(column))
        except (TypeError, ValueError):
            years.append(i)
    return years


def dataframe_to_series(df: pd.DataFrame, name: str) -> pd.Series:
    out = df.copy()
    out.columns = _normalise_weather_years(out.columns)
    out = out.astype("float32", copy=False)
    try:
        series = out.stack(future_stack=True).rename(name)
    except TypeError:
        series = out.stack(dropna=False).rename(name)
    series.index = series.index.set_names(["timestamp", "weather_year"])
    return series


def make_timeseries_records(
    *,
    record_type: str,
    area: str,
    entity: str,
    object_name: str = "",
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    series = [dataframe_to_series(df, name) for name, df in frames.items() if df is not None]
    if not series:
        return pd.DataFrame(columns=PROCESSED_COLUMNS)

    out = pd.concat(series, axis=1).reset_index()
    out["schema_version"] = SCHEMA_VERSION
    out["record_type"] = record_type
    out["area"] = area
    out["entity"] = entity
    out["object_name"] = object_name
    return normalise_processed_frame(out)


def normalise_processed_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in PROCESSED_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    out = out[PROCESSED_COLUMNS]
    out["schema_version"] = out["schema_version"].fillna(SCHEMA_VERSION).astype("int16")
    out["weather_year"] = pd.to_numeric(out["weather_year"], errors="coerce").fillna(-1).astype("int16")
    for column in STRING_COLUMNS:
        out[column] = out[column].fillna("").astype(str)
    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    return out


def write_metadata(
    *,
    metadata_path: Path,
    model_folder: str,
    scenario_name: str,
    source_result_path: Path,
    processed_data_path: Path,
    options: dict,
    row_counts: dict[str, int],
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_folder": model_folder,
        "scenario_name": scenario_name,
        "source_result_path": str(source_result_path),
        "processed_data_path": str(processed_data_path),
        "options": options,
        "row_counts": row_counts,
        "columns": PROCESSED_COLUMNS,
    }
    with open(metadata_path, "w") as f:
        json.dump(payload, f, indent=2)


@dataclass
class ProcessedScenarioResults:
    data_path: Path
    result_path: Path | None = None

    @classmethod
    def from_result_path(
        cls,
        result_path: Path,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ) -> "ProcessedScenarioResults | None":
        data_path = processed_data_path_for_result(result_path, output_root)
        if not data_path.exists():
            return None
        return cls(data_path=data_path, result_path=Path(result_path))

    @property
    def name(self) -> str:
        if self.result_path is not None:
            return Path(self.result_path).name
        return self.data_path.parent.name

    def _read(
        self,
        *,
        filters: list[tuple[str, str, object]],
        columns: list[str],
    ) -> pd.DataFrame:
        return pd.read_parquet(self.data_path, columns=columns, filters=filters)

    def _matrix(
        self,
        *,
        record_type: str,
        entity: str,
        value_column: str,
        area: str | None = None,
        object_name: str | None = None,
    ) -> pd.DataFrame:
        filters: list[tuple[str, str, object]] = [
            ("record_type", "==", record_type),
            ("entity", "==", entity),
        ]
        if area is not None:
            filters.append(("area", "==", area))
        if object_name is not None:
            filters.append(("object_name", "==", object_name))

        try:
            df = self._read(
                filters=filters,
                columns=["timestamp", "weather_year", value_column],
            ).dropna(subset=[value_column])
        except Exception as exc:
            raise KeyError(f"No processed {record_type}/{entity}/{value_column} in {self.data_path}") from exc
        if df.empty:
            raise KeyError(f"No processed {record_type}/{entity}/{value_column} in {self.data_path}")

        matrix = df.pivot_table(
            index="timestamp",
            columns="weather_year",
            values=value_column,
            aggfunc="first",
        )
        matrix = matrix.sort_index().sort_index(axis=1)
        matrix.columns.name = None
        return matrix.astype("float32", copy=False)

    def _unique(self, *, record_type: str, column: str) -> list[str]:
        df = self._read(
            filters=[("record_type", "==", record_type)],
            columns=[column],
        )
        return sorted(str(x) for x in df[column].dropna().unique() if str(x))

    def get_busbar_names(self) -> list[str]:
        return self._unique(record_type="busbar", column="entity")

    def get_dcline_names(self) -> list[str]:
        return self._unique(record_type="dcline", column="entity")

    def get_reservoir_entities(self) -> list[str]:
        return self._unique(record_type="reservoir", column="entity")

    def get_prices_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="price")

    def get_load_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="load")

    def get_hydro_production_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="hydro")

    def get_reservoir_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="reservoir")

    def get_market_steps_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="market_steps")

    def get_solar_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="solar")

    def get_onshore_wind_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="onshore_wind")

    def get_offshore_wind_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="offshore_wind")

    def get_fixed_nuclear_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="fixed_nuclear")

    def get_historic_nuclear_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="historic_nuclear")

    def get_historic_nuclear_available_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="historic_nuclear_available")

    def get_new_nuclear_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="new_nuclear")

    def get_new_nuclear_available_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="new_nuclear_available")

    def get_total_nuclear_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="total_nuclear")

    def get_total_nuclear_available_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="total_nuclear_available")

    def get_biomass_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="biomass")

    def get_fossil_gas_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="fossil_gas")

    def get_fossil_other_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="fossil_other")

    def get_rationing_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="rationing")

    def get_market_spillage_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="market_spillage")

    def get_reservoir_spill_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="reservoir_spill")

    def get_reservoir_discharge_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._matrix(record_type="busbar", entity=busbar_name, value_column="reservoir_discharge")

    def get_busbar_metric(self, busbar_name: str, metric: str) -> pd.DataFrame:
        if metric not in NUMERIC_COLUMNS:
            raise KeyError(f"Unknown processed metric: {metric}")
        return self._matrix(record_type="busbar", entity=busbar_name, value_column=metric)

    def get_dcline_flow(self, dcline_name: str) -> pd.DataFrame:
        return self._matrix(record_type="dcline", entity=dcline_name, value_column="flow")

    def get_reservoir_metric(self, entity: str, metric: str) -> pd.DataFrame:
        if metric not in {
            "reservoir_production",
            "reservoir_level",
            "reservoir_spill",
            "reservoir_discharge",
        }:
            raise KeyError(f"Unsupported reservoir metric: {metric}")
        return self._matrix(record_type="reservoir", entity=entity, value_column=metric)

    def get_reservoir_production(self, entity: str) -> pd.DataFrame:
        return self.get_reservoir_metric(entity, "reservoir_production")


def add_project_paths(project_root: Path) -> None:
    for path in (project_root, project_root / "data"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)
