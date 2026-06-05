from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from lpr_sintef_bifrost.inputs import TimeSeries
from lpr_sintef_bifrost.inputs.timeseries import ConstantTimeseriesConfig
from lpr_sintef_bifrost.models.common import MarketStep
from lpr_sintef_bifrost.models.emps import Wind
from lpr_sintef_bifrost.utils.unit import Unit

HISTORIC_NUCLEAR_PROFILE_PATH = Path("data/historic_nuclear_profile.parquet")
NEW_NUCLEAR_PROFILE_PATH = Path("data/new_nuclear_profile.parquet")
IMPROVE_NUCLEAR_REP = False
NEW_NUCLEAR_FIRM_SHARE = 0.60
NEW_NUCLEAR_MARGINAL_COST = 9.0

HISTORIC_NUCLEAR_PREFIX = "historic_nuclear"
HISTORIC_NUCLEAR_FLEXIBLE_PREFIX = "historic_nuclear_flexible"
NEW_NUCLEAR_FIRM_PREFIX = "new_nuclear_firm"
NEW_NUCLEAR_FLEXIBLE_PREFIX = "new_nuclear_flexible"

_logger = logging.getLogger("lpr_sintef_bifrost")


def improved_nuclear_output_suffix(improve_nuclear_rep: bool) -> str:
    return "_imp_nuke" if improve_nuclear_rep else ""


def build_scenario_profile(
    profile: pd.DataFrame,
    simulation_time_index: pd.DatetimeIndex,
    start_scenario_year: int,
    end_scenario_year: int,
    fallback_value: float,
) -> pd.DataFrame:
    if profile.empty:
        series = pd.Series(fallback_value, index=simulation_time_index)
    else:
        series = pd.to_numeric(profile.iloc[:, 0], errors="coerce").ffill().bfill().fillna(fallback_value)

    def values_for_scenario(scenario_year: int) -> np.ndarray:
        values = series
        if isinstance(series.index, pd.DatetimeIndex):
            scenario_values = series.loc[series.index.year >= scenario_year]
            if not scenario_values.empty:
                values = scenario_values

        data = values.to_numpy()
        if len(data) == 0:
            data = np.array([fallback_value])
        if len(data) < len(simulation_time_index):
            repeats = int(np.ceil(len(simulation_time_index) / len(data)))
            data = np.tile(data, repeats)
        return data[: len(simulation_time_index)]

    data = [values_for_scenario(year) for year in range(start_scenario_year, end_scenario_year + 1)]
    return pd.DataFrame(
        index=simulation_time_index,
        data=np.array(data).T,
        columns=range(start_scenario_year, end_scenario_year + 1),
    )


def nuclear_capacity_profile(
    *,
    profile_path: Path,
    simulation_time_index: pd.DatetimeIndex,
    start_scenario_year: int,
    end_scenario_year: int,
    capacity: float,
    fallback_capacity_factor: float = 0.90,
) -> pd.DataFrame:
    profile = pd.read_parquet(profile_path)
    return (
        build_scenario_profile(
            profile=profile,
            simulation_time_index=simulation_time_index,
            start_scenario_year=start_scenario_year,
            end_scenario_year=end_scenario_year,
            fallback_value=fallback_capacity_factor,
        )
        * capacity
    )


def get_busbar(config, node: str):
    for busbar in config.busbars:
        if busbar.name == node:
            return busbar
    raise KeyError(f"Busbar {node} not found")


def _timeseries(df: pd.DataFrame) -> TimeSeries:
    return TimeSeries(
        value=df.copy(),
        unit=Unit.MW,
        enforce_scenario_dimensions=True,
    )


def _market_step_timeseries(df: pd.DataFrame) -> TimeSeries:
    """MarketStep capacity ("mengde") supports a single deterministic profile only."""
    value = df.copy()
    if isinstance(value, pd.DataFrame) and value.shape[1] > 1:
        first = value.iloc[:, 0]
        for column in value.columns[1:]:
            if not np.allclose(first.to_numpy(), value[column].to_numpy(), equal_nan=True):
                _logger.warning(
                    "Collapsing non-identical MarketStep capacity scenarios to the first profile. "
                    "PyLTM does not support multi-scenario market-step capacity."
                )
                break
        value = first.to_frame(name="capacity")

    return TimeSeries(
        value=value,
        unit=Unit.MW,
        enforce_scenario_dimensions=True,
    )


def _add_wind(config, *, node: str, name: str, capacity: pd.DataFrame) -> Wind:
    nuclear = Wind(name=name, capacity=_timeseries(capacity))
    config.add(nuclear)
    config.connect(to_obj=get_busbar(config, node), from_obj=nuclear)
    return nuclear


def _add_market_step(config, *, node: str, name: str, capacity: pd.DataFrame, price: float) -> MarketStep:
    nuclear = MarketStep(
        name=name,
        capacity=_market_step_timeseries(capacity),
        price=TimeSeries(config=ConstantTimeseriesConfig(value=price), unit=Unit.EUR_MWH),
        fuel_type="nuclear",
    )
    config.add(nuclear)
    config.connect(to_obj=get_busbar(config, node), from_obj=nuclear)
    return nuclear


def add_historic_nuclear(
    *,
    config,
    node: str,
    capacity: float,
    profile_path: Path = HISTORIC_NUCLEAR_PROFILE_PATH,
    fallback_capacity_factor: float = 1.0,
    logger: logging.Logger | None = None,
) -> Wind:
    logger = logger or _logger
    simulation_time_index = pd.date_range(start=config.start, end=config.end, freq="1h")
    capacity_profile = nuclear_capacity_profile(
        profile_path=profile_path,
        simulation_time_index=simulation_time_index,
        start_scenario_year=config.start_scenario_year,
        end_scenario_year=config.end_scenario_year,
        capacity=capacity,
        fallback_capacity_factor=fallback_capacity_factor,
    )
    name = f"{HISTORIC_NUCLEAR_PREFIX}_{node}_{uuid.uuid4().hex[:4]}"
    logger.info("Adding historic firm nuclear for %s. %.2f MW installed capacity.", node, capacity)
    return _add_wind(config, node=node, name=name, capacity=capacity_profile)


def add_historic_flexible_nuclear(
    *,
    config,
    node: str,
    capacity: float,
    price: float = NEW_NUCLEAR_MARGINAL_COST,
    profile_path: Path = NEW_NUCLEAR_PROFILE_PATH,
    fallback_capacity_factor: float = 0.90,
    logger: logging.Logger | None = None,
) -> MarketStep:
    logger = logger or _logger
    simulation_time_index = pd.date_range(start=config.start, end=config.end, freq="1h")
    capacity_profile = nuclear_capacity_profile(
        profile_path=profile_path,
        simulation_time_index=simulation_time_index,
        start_scenario_year=config.start_scenario_year,
        end_scenario_year=config.end_scenario_year,
        capacity=capacity,
        fallback_capacity_factor=fallback_capacity_factor,
    )
    name = f"{HISTORIC_NUCLEAR_FLEXIBLE_PREFIX}_{node}_{uuid.uuid4().hex[:4]}"
    logger.info(
        "Adding historic flexible nuclear for %s. %.2f MW installed capacity at %.2f EUR/MWh.",
        node,
        capacity,
        price,
    )
    return _add_market_step(config, node=node, name=name, capacity=capacity_profile, price=price)


def add_new_nuclear(
    *,
    config,
    node: str,
    capacity: float,
    price: float = NEW_NUCLEAR_MARGINAL_COST,
    improve_nuclear_rep: bool = False,
    firm_share: float = NEW_NUCLEAR_FIRM_SHARE,
    profile_path: Path = NEW_NUCLEAR_PROFILE_PATH,
    fallback_capacity_factor: float = 0.90,
    logger: logging.Logger | None = None,
) -> list[Wind | MarketStep]:
    logger = logger or _logger
    firm_share = firm_share if improve_nuclear_rep else 0.0
    firm_share = min(max(float(firm_share), 0.0), 1.0)
    flexible_share = 1.0 - firm_share

    simulation_time_index = pd.date_range(start=config.start, end=config.end, freq="1h")
    total_capacity_profile = nuclear_capacity_profile(
        profile_path=profile_path,
        simulation_time_index=simulation_time_index,
        start_scenario_year=config.start_scenario_year,
        end_scenario_year=config.end_scenario_year,
        capacity=capacity,
        fallback_capacity_factor=fallback_capacity_factor,
    )

    objects: list[Wind | MarketStep] = []
    suffix = uuid.uuid4().hex[:4]

    if firm_share > 0:
        firm_name = f"{NEW_NUCLEAR_FIRM_PREFIX}_{node}_{suffix}"
        firm_capacity = total_capacity_profile * firm_share
        logger.info(
            "Adding new firm nuclear for %s. %.2f MW installed capacity, %.1f%% firm share.",
            node,
            capacity,
            firm_share * 100,
        )
        objects.append(_add_wind(config, node=node, name=firm_name, capacity=firm_capacity))

    if flexible_share > 0:
        flexible_name = f"{NEW_NUCLEAR_FLEXIBLE_PREFIX}_{node}_{suffix}"
        flexible_capacity = total_capacity_profile * flexible_share
        logger.info(
            "Adding new flexible nuclear for %s. %.2f MW installed capacity, %.1f%% flexible share at %.2f EUR/MWh.",
            node,
            capacity,
            flexible_share * 100,
            price,
        )
        objects.append(
            _add_market_step(
                config,
                node=node,
                name=flexible_name,
                capacity=flexible_capacity,
                price=price,
            )
        )

    return objects
