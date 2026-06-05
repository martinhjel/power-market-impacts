from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lpr_sintef_bifrost.utils.time import CET_winter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data import PowerGamaDataLoader
from dataset_runner import LoadMode
from scenario_runner import SCENARIOS, ScenarioConfig, select_scenarios

GENERATION_COLUMNS = ("solar", "wind", "nuclear")
NORWAY_AREAS = {f"NO{i}" for i in range(1, 6)}
MARKET_CALIBRATION_AREAS = {
    "NO_South": ["NO1", "NO2", "NO5", "SE3", "SE4"],
    "NO_North": ["SE1", "SE2", "NO3", "NO4"],
}
VALIDATION_MARKET_CALIBRATION_AREAS = ("NO_South", "NO_North")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationFailure:
    scenario_name: str
    area: str
    weather_scenario: int
    iso_year: int
    iso_week: int
    generation_mwh: float
    generation_wo_nuclear_mwh: float
    load_mwh: float

    @property
    def surplus_mwh(self) -> float:
        return self.generation_mwh - self.load_mwh

    @property
    def excess_pct_of_load(self) -> float:
        if self.load_mwh == 0:
            return float("inf")
        return self.surplus_mwh / self.load_mwh * 100.0

    @property
    def surplus_wo_nuclear_mwh(self) -> float:
        return self.generation_wo_nuclear_mwh - self.load_mwh

    @property
    def excess_wo_nuclear_pct_of_load(self) -> float:
        if self.load_mwh == 0:
            return float("inf")
        return self.surplus_wo_nuclear_mwh / self.load_mwh * 100.0


def simulation_index(simulation_years: int = 1) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=2024, month=1, day=1, hour=0, minute=0, second=0, tz=CET_winter)
    end = start + pd.Timedelta(weeks=52 * simulation_years)
    return pd.date_range(start=start, end=end, freq="1h")


def scenario_columns(start_scenario_year: int, end_scenario_year: int) -> range:
    return range(start_scenario_year, end_scenario_year + 1)


def as_scenario_profile(
    profile: pd.DataFrame | pd.Series,
    index: pd.DatetimeIndex,
    start_scenario_year: int,
    end_scenario_year: int,
    fallback_value: float = 0.0,
) -> pd.DataFrame:
    if isinstance(profile, pd.DataFrame):
        series = profile.iloc[:, 0] if not profile.empty else pd.Series(dtype=float)
    else:
        series = profile
    series = pd.to_numeric(series, errors="coerce").ffill().bfill().fillna(fallback_value)

    if series.empty:
        series = pd.Series(fallback_value, index=index)

    def values_for_scenario(year: int) -> np.ndarray:
        values = series
        if isinstance(series.index, pd.DatetimeIndex):
            scenario_values = series.loc[(series.index.year >= year)]
            if not scenario_values.empty:
                values = scenario_values

        data = values.to_numpy()
        if len(data) == 0:
            data = np.array([fallback_value])
        if len(data) < len(index):
            repeats = int(np.ceil(len(index) / len(data)))
            data = np.tile(data, repeats)
        return data[: len(index)]

    data = [values_for_scenario(year) for year in scenario_columns(start_scenario_year, end_scenario_year)]
    return pd.DataFrame(index=index, data=np.array(data).T, columns=scenario_columns(start_scenario_year, end_scenario_year))


def empty_timeseries(index: pd.DatetimeIndex, start_scenario_year: int, end_scenario_year: int) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=scenario_columns(start_scenario_year, end_scenario_year))


def add_to_area(target: dict[str, pd.DataFrame], area: str, values: pd.DataFrame) -> None:
    if area in target:
        target[area] = target[area].add(values, fill_value=0.0)
    else:
        target[area] = values.copy()


def load_powergama_data(args: argparse.Namespace) -> PowerGamaDataLoader:
    loader = PowerGamaDataLoader(
        year=args.dataset_year,
        scenario=args.dataset_scenario,
        version=args.dataset_version,
        base_path=args.base_path,
        combined=args.combined,
    )
    generator_override = Path("data/generator.csv")
    if generator_override.exists():
        loader.generator = pd.read_csv(generator_override, index_col=0)
    return loader


def load_node_for_profile(node: str) -> str:
    if node in ["EE", "FI", "LT"]:
        return "FIN"
    if node == "GB":
        return "GBR"
    if node == "DE":
        return "DEU"
    if node == "NL":
        return "NLD"
    if node == "PL":
        return "DEU"
    return node


def build_base_loads(
    loader: PowerGamaDataLoader,
    index: pd.DatetimeIndex,
    start_scenario_year: int,
    end_scenario_year: int,
) -> dict[str, pd.DataFrame]:
    loads: dict[str, pd.DataFrame] = {}
    for _, row in loader.consumer.iterrows():
        profile_node = load_node_for_profile(row["node"])
        df_load = pd.read_csv(
            f"data/Profiler/Consumption/{profile_node}_consumption.csv",
            index_col=0,
            parse_dates=True,
        )
        df_load = df_load.loc[
            (df_load.index.year >= start_scenario_year) & (df_load.index.year <= end_scenario_year)
        ]

        data = []
        for year in scenario_columns(start_scenario_year, end_scenario_year):
            scenario_values = df_load.loc[df_load.index.year >= year].iloc[: len(index)].to_numpy().squeeze()
            data.append(scenario_values)

        df = pd.DataFrame(index=index, data=np.array(data).T, columns=scenario_columns(start_scenario_year, end_scenario_year))
        loads[row["node"]] = df / df.mean() * row["demand_avg"]
    return loads


def apply_scenario_load(loads: dict[str, pd.DataFrame], scenario: ScenarioConfig) -> dict[str, pd.DataFrame]:
    loads = {area: df.copy() for area, df in loads.items()}
    if scenario.additional_load_twh <= 0 or scenario.load_mode == LoadMode.NONE:
        return loads

    norwegian_loads = {area: df for area, df in loads.items() if area in NORWAY_AREAS}
    total_mean = sum(df.mean().mean() for df in norwegian_loads.values())
    if total_mean <= 0:
        return loads

    template = next(iter(norwegian_loads.values()))
    timestep_hours = (template.index[1] - template.index[0]).total_seconds() / 3600.0 if len(template.index) > 1 else 1.0
    total_hours = len(template.index) * timestep_hours
    avg_new_demand_mw = scenario.additional_load_twh * 1e6 / total_hours

    for area, df in norwegian_loads.items():
        base_mean = df.mean().mean()
        if base_mean <= 0:
            continue
        share = base_mean / total_mean
        additional_avg_mw = share * avg_new_demand_mw
        if scenario.load_mode == LoadMode.LLPS:
            loads[area] = df * (1.0 + additional_avg_mw / base_mean)
        elif scenario.load_mode == LoadMode.BA:
            loads[area] = df + additional_avg_mw
    return loads


def build_base_generation(
    loader: PowerGamaDataLoader,
    renewables_profiles: pd.DataFrame,
    index: pd.DatetimeIndex,
    start_scenario_year: int,
    end_scenario_year: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    generation = {column: {} for column in GENERATION_COLUMNS}
    historic_nuclear_profile = pd.read_parquet("data/historic_nuclear_profile.parquet")

    for _, row in loader.generator.iterrows():
        area = row["node"]
        generator_type = row["type"]
        pmax = float(row["pmax"])
        if pmax <= 0:
            continue

        if generator_type in ["wind_off", "wind_on", "solar"]:
            inflow_ref = row["inflow_ref"]
            if inflow_ref not in renewables_profiles.columns:
                continue
            profile = renewables_profiles.loc[:, inflow_ref] * pmax
            capacity = as_scenario_profile(profile, index, start_scenario_year, end_scenario_year)
            key = "solar" if generator_type == "solar" else "wind"
            add_to_area(generation[key], area, capacity)
        elif generator_type == "nuclear":
            capacity = as_scenario_profile(
                historic_nuclear_profile,
                index,
                start_scenario_year,
                end_scenario_year,
                fallback_value=1.0,
            ) * pmax
            add_to_area(generation["nuclear"], area, capacity)

    return generation


def offshore_target_area(offshore: dict, idx: int, map_to_connected: bool) -> str:
    if map_to_connected:
        return offshore["connected_to"]
    return offshore.get("new_area_name") or offshore.get("new_area") or f"{offshore['connected_to']}_offshore_{idx}"


def apply_scenario_generation(
    generation: dict[str, dict[str, pd.DataFrame]],
    scenario: ScenarioConfig,
    renewables_profiles: pd.DataFrame,
    index: pd.DatetimeIndex,
    start_scenario_year: int,
    end_scenario_year: int,
    map_offshore_to_connected: bool,
) -> dict[str, dict[str, pd.DataFrame]]:
    generation = {
        technology: {area: df.copy() for area, df in area_values.items()}
        for technology, area_values in generation.items()
    }

    if scenario.nuclear_additions:
        nuclear_profile = pd.read_parquet("data/new_nuclear_profile.parquet")
        for nuclear in scenario.nuclear_additions:
            capacity = as_scenario_profile(
                nuclear_profile,
                index,
                start_scenario_year,
                end_scenario_year,
                fallback_value=0.90,
            ) * float(nuclear["capacity"])
            add_to_area(generation["nuclear"], nuclear["area"], capacity)

    for idx, offshore in enumerate(scenario.offshore_wind_additions, start=1):
        profile_name = offshore["profile"]
        if profile_name not in renewables_profiles.columns:
            raise ValueError(f"Missing offshore wind profile: {profile_name}")
        area = offshore_target_area(offshore, idx=idx, map_to_connected=map_offshore_to_connected)
        pmax = float(offshore.get("pmax", offshore["capacity"]))
        capacity = as_scenario_profile(
            renewables_profiles.loc[:, profile_name] * pmax,
            index,
            start_scenario_year,
            end_scenario_year,
        )
        add_to_area(generation["wind"], area, capacity)

    return generation


def total_generation_by_area(
    generation: dict[str, dict[str, pd.DataFrame]],
    technologies: tuple[str, ...] = GENERATION_COLUMNS,
) -> dict[str, pd.DataFrame]:
    totals: dict[str, pd.DataFrame] = {}
    for technology in technologies:
        area_values = generation[technology]
        for area, df in area_values.items():
            add_to_area(totals, area, df)
    return totals


def aggregate_market_calibration_areas(
    values_by_area: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    start_scenario_year: int,
    end_scenario_year: int,
) -> dict[str, pd.DataFrame]:
    aggregated = {}
    zero = empty_timeseries(index, start_scenario_year, end_scenario_year)
    for market_area in VALIDATION_MARKET_CALIBRATION_AREAS:
        total = zero.copy()
        for area in MARKET_CALIBRATION_AREAS[market_area]:
            total = total.add(values_by_area.get(area, zero), fill_value=0.0)
        aggregated[market_area] = total
    return aggregated


def weekly_sums(df: pd.DataFrame) -> pd.DataFrame:
    long = df.stack().rename("value").reset_index()
    long.columns = ["timestamp", "weather_scenario", "value"]
    iso = long["timestamp"].dt.isocalendar()
    long["iso_year"] = iso.year.astype(int)
    long["iso_week"] = iso.week.astype(int)
    return long.groupby(["weather_scenario", "iso_year", "iso_week"], dropna=False)["value"].sum().to_frame()


def validate_scenario_inputs(
    scenario: ScenarioConfig,
    base_loads: dict[str, pd.DataFrame],
    base_generation: dict[str, dict[str, pd.DataFrame]],
    renewables_profiles: pd.DataFrame,
    index: pd.DatetimeIndex,
    start_scenario_year: int,
    end_scenario_year: int,
    map_offshore_to_connected: bool,
    tolerance_mwh: float,
) -> list[ValidationFailure]:
    loads = apply_scenario_load(base_loads, scenario)
    generation = apply_scenario_generation(
        base_generation,
        scenario,
        renewables_profiles,
        index,
        start_scenario_year,
        end_scenario_year,
        map_offshore_to_connected,
    )
    generation_totals = total_generation_by_area(generation)
    generation_wo_nuclear_totals = total_generation_by_area(generation, technologies=("solar", "wind"))
    market_area_loads = aggregate_market_calibration_areas(
        loads,
        index,
        start_scenario_year,
        end_scenario_year,
    )
    market_area_generation = aggregate_market_calibration_areas(
        generation_totals,
        index,
        start_scenario_year,
        end_scenario_year,
    )
    market_area_generation_wo_nuclear = aggregate_market_calibration_areas(
        generation_wo_nuclear_totals,
        index,
        start_scenario_year,
        end_scenario_year,
    )

    failures: list[ValidationFailure] = []
    for market_area in VALIDATION_MARKET_CALIBRATION_AREAS:
        weekly_load = weekly_sums(market_area_loads[market_area])
        weekly_generation = weekly_sums(market_area_generation[market_area])
        weekly_generation_wo_nuclear = weekly_sums(market_area_generation_wo_nuclear[market_area])
        weekly = weekly_generation.rename(columns={"value": "generation_mwh"}).join(
            weekly_load.rename(columns={"value": "load_mwh"}),
            how="outer",
        ).fillna(0.0)
        weekly = weekly.join(
            weekly_generation_wo_nuclear.rename(columns={"value": "generation_wo_nuclear_mwh"}),
            how="outer",
        ).fillna(0.0)
        weekly["surplus_mwh"] = weekly["generation_mwh"] - weekly["load_mwh"]
        weekly = weekly[weekly["surplus_mwh"] > tolerance_mwh]

        for (weather_scenario, iso_year, iso_week), row in weekly.iterrows():
            failures.append(
                ValidationFailure(
                    scenario_name=scenario.name,
                    area=market_area,
                    weather_scenario=int(weather_scenario),
                    iso_year=int(iso_year),
                    iso_week=int(iso_week),
                    generation_mwh=float(row["generation_mwh"]),
                    generation_wo_nuclear_mwh=float(row["generation_wo_nuclear_mwh"]),
                    load_mwh=float(row["load_mwh"]),
                )
            )
    return failures


def log_failures(failures: list[ValidationFailure], max_rows: int) -> None:
    if max_rows <= 0:
        return

    for idx, failure in enumerate(failures):
        if idx >= max_rows:
            logger.warning("... %s more validation failures not shown", len(failures) - max_rows)
            break
        logger.warning(
            "FAIL "
            "sc=%s "
            "mca=%s "
            "wy=%s "
            "wk=%s-W%02d "
            "gen=%sTWh "
            "load=%sTWh "
            "ex=%sTWh/%s%% "
            "wo_nuke=%sTWh "
            "wo_fail=%s "
            "wo_ex=%sTWh/%s%% "
            "nuke=%sTWh",
            failure.scenario_name,
            failure.area,
            failure.weather_scenario,
            failure.iso_year,
            failure.iso_week,
            f"{failure.generation_mwh / 1e6:,.4f}",
            f"{failure.load_mwh / 1e6:,.4f}",
            f"{failure.surplus_mwh / 1e6:,.4f}",
            f"{failure.excess_pct_of_load:,.2f}",
            f"{failure.generation_wo_nuclear_mwh / 1e6:,.4f}",
            failure.surplus_wo_nuclear_mwh > 0.0,
            f"{failure.surplus_wo_nuclear_mwh / 1e6:,.4f}",
            f"{failure.excess_wo_nuclear_pct_of_load:,.2f}",
            f"{(failure.generation_mwh - failure.generation_wo_nuclear_mwh) / 1e6:,.4f}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-run validation that weekly solar + wind + nuclear input generation does not exceed "
            "weekly load for each area and weather scenario."
        )
    )
    parser.add_argument("--only", nargs="+", help="Validate only these scenario_runner scenario names.")
    parser.add_argument("--dataset-year", type=int, default=2025)
    parser.add_argument("--dataset-scenario", default="BM")
    parser.add_argument("--dataset-version", default="100")
    parser.add_argument("--base-path", type=Path, default=Path.cwd() / "data/NordicNuclearAnalysis")
    parser.add_argument("--start-scenario-year", type=int, default=1991)
    parser.add_argument("--end-scenario-year", type=int, default=2020)
    parser.add_argument("--combined", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--own-offshore-areas",
        action="store_true",
        help=(
            "Keep offshore wind in its created offshore busbar. By default it is counted against "
            "the connected_to area from scenario_runner.py."
        ),
    )
    parser.add_argument("--tolerance-mwh", type=float, default=1e-6)
    parser.add_argument("--max-rows", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = select_scenarios(args.only) if args.only else SCENARIOS
    index = simulation_index()

    loader = load_powergama_data(args)
    renewables_profiles = pd.read_parquet("data/renewables_profiles.parquet")
    base_loads = build_base_loads(loader, index, args.start_scenario_year, args.end_scenario_year)
    base_generation = build_base_generation(
        loader,
        renewables_profiles,
        index,
        args.start_scenario_year,
        args.end_scenario_year,
    )

    all_failures: list[ValidationFailure] = []
    for scenario in scenarios:
        logger.info("Validating scenario %s", scenario.name)
        failures = validate_scenario_inputs(
            scenario,
            base_loads,
            base_generation,
            renewables_profiles,
            index,
            args.start_scenario_year,
            args.end_scenario_year,
            map_offshore_to_connected=not args.own_offshore_areas,
            tolerance_mwh=args.tolerance_mwh,
        )
        all_failures.extend(failures)
        if failures:
            log_failures(failures, max_rows=max(args.max_rows - (len(all_failures) - len(failures)), 0))
            logger.warning("Scenario %s failed: %s weekly area checks exceeded load.", scenario.name, len(failures))
        else:
            logger.info("Scenario %s passed.", scenario.name)

    if all_failures:
        if len(all_failures) > args.max_rows > 0:
            logger.warning("Total hidden validation failures: %s", len(all_failures) - args.max_rows)
        logger.error("Validation failed: %s weekly area checks exceeded load.", len(all_failures))
        return 1

    logger.info("Validation passed: checked %s scenarios.", len(scenarios))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
