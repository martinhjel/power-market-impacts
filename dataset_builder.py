# %%
import logging
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from cognite.client.testing import CogniteClientMock
from lpr_sintef_bifrost.filter import ModelFilter
from lpr_sintef_bifrost.inputs import TimeSeries
from lpr_sintef_bifrost.inputs.timeseries import ConstantTimeseriesConfig
from lpr_sintef_bifrost.models import EMPSModelBuilder
from lpr_sintef_bifrost.models._model import ModelMetadata
from lpr_sintef_bifrost.models.common import (
    Busbar,
    InflowSeries,
    Load,
    MarketStep,
    PriceSeries,
)
from lpr_sintef_bifrost.models.emps import DcLine, Solar, Wind
from lpr_sintef_bifrost.utils.time import CET_winter
from lpr_sintef_bifrost.utils.unit import Unit

from data import PowerGamaDataLoader
from nuclear_modeling import (
    HISTORIC_NUCLEAR_PROFILE_PATH,
    IMPROVE_NUCLEAR_REP as DEFAULT_IMPROVE_NUCLEAR_REP,
    NEW_NUCLEAR_FIRM_SHARE,
    NEW_NUCLEAR_MARGINAL_COST,
    NEW_NUCLEAR_PROFILE_PATH,
    add_historic_flexible_nuclear,
    add_historic_nuclear,
    improved_nuclear_output_suffix,
)

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger("lpr_sintef_bifrost")

model_name = "PowerGamaMSc"
use_exogenous_prices = True

resolution = "1H"
clear_destination_folder = True
dry_run = False
verbose = False
convert_large_timeseries_to_hdf5 = True
export_detailed_results = False
cdf_client = CogniteClientMock()
# client = Clients.default(client_name="bifrost-cli-runner", env="local")

simulation_start = pd.Timestamp(year=2024, month=1, day=1, hour=0, minute=0, second=0, tz=CET_winter)
simulation_years = 1
simulation_end = simulation_start + pd.Timedelta(weeks=52 * simulation_years)

start_scenario_year = 1991
end_scenario_year = 2020
start_reservoir_value_pu = 0.6587  # Historical mean for Norway
simulation_type = "serial"  # ["parallel", "serial"]

dataset_year = 2025
dataset_scenario = "BM"
dataset_version = "100"
base_path = Path.cwd() / "data/NordicNuclearAnalysis"
combined = True
dataset_path = base_path / f"CASE_{dataset_year}/scenario_{dataset_scenario}/data/system"

load_per_scenario = True
IMPROVE_NUCLEAR_REP = True

destination_folder = (
    Path.cwd()
    / (
        f"ltm_output/{model_name}_{dataset_year}_{dataset_scenario}_{resolution}_"
        f"{simulation_type}_{use_exogenous_prices}EXO_load"
        f"{improved_nuclear_output_suffix(IMPROVE_NUCLEAR_REP)}/"
    )
)
destination_folder.mkdir(exist_ok=True, parents=True)

file_handler = logging.FileHandler(str(destination_folder / "log.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_logger.addHandler(file_handler)

data_loader = PowerGamaDataLoader(
    year=dataset_year, scenario=dataset_scenario, version=dataset_version, base_path=base_path, combined=combined
)
# NB: Temporary overwriting of generator data
generator = pd.read_csv("data/generator.csv", index_col=0)
data_loader.generator = generator

_logger.info("EMPS PowerGama & NVE dataset")

# %%

profiles_file = Path.cwd() / "data/Profiler/timeseries_profiles_v3.csv"
df_profiles = pd.read_csv(profiles_file, index_col=0, parse_dates=True)

df_renewables_profiles = pd.read_parquet("data/renewables_profiles.parquet")
df_load_profiles = pd.read_parquet("data/load_profiles.parquet")

db_profiles = set(list(df_renewables_profiles.columns) + list(df_load_profiles.columns))
gen_profiles = set(data_loader.generator["inflow_ref"].unique())

missing_profiles = gen_profiles - db_profiles
_logger.warning(f"Missing profiles {missing_profiles}")


# %%

config = EMPSModelBuilder(
    start=simulation_start,
    end=simulation_end,
    simulation_start=simulation_start,
    start_scenario_year=start_scenario_year,
    end_scenario_year=end_scenario_year,
    cdf_client=cdf_client,
)

config.global_settings.simulation_type = simulation_type

# Start simulation from the current ISO week
# config.global_settings.simulation_period.config.start_from_current_iso_week = True

config.global_settings.intraweek_timesteps = None
# Parse the resolution string like "1H", "3D", etc.
int_val = int(resolution[:-1])  # e.g. "1H" -> 1
freq = resolution[-1]  # e.g. "1H" -> "H"

if freq == "H":
    timesteps_per_week = int(7 * 24 / int_val)
elif freq == "D":
    timesteps_per_week = int(7 / int_val)
else:
    raise ValueError(f"Unsupported resolution: {resolution}")

config.global_settings.timesteps_per_week = timesteps_per_week
# config.global_settings.simulation_period = TimeSeries()

############# Add PowerGama data ##############
# %%  Busbars

market_calibration_areas = {
    # "DE": ["DE"],
    "EE": ["EE"],
    "FI": ["FI"],
    # "GB": ["GB"],
    "LT": ["LT"],
    # "NL": ["NL"],
    "NO_South": ["NO1", "NO2", "NO5", "SE3", "SE4"],
    "NO_North": ["SE1", "SE2", "NO3", "NO4"],
    # "NO": ["NO1", "NO2", "NO3", "NO4", "NO5", "DK1", "DK2"],
    # "NO": ["NO1", "NO2", "NO3", "NO4", "NO5"],
    # "PL": ["PL"],
    "DK": ["DK1", "DK2"],
    # "SE": ["SE1", "SE2", "SE3", "SE4"],
}


exogenous_busbars = []
if use_exogenous_prices:
    exogenous_busbars = ["DE", "GB", "PL", "NL"]

aggregated_hydropower_busbars = ["EE", "LT"]

# %%
busbars = []
for market_calibration_area in market_calibration_areas:
    busbars += market_calibration_areas[market_calibration_area]

busbars += exogenous_busbars

busbars_obj = {}
for busbar in busbars:
    busbars_obj[busbar] = Busbar(name=busbar)
    config.add(busbars_obj[busbar])

# Add market calibration areas
_logger.info(f"Adding market calibration areas: {market_calibration_areas}")
_market_calibration_areas = market_calibration_areas.copy()
_market_calibration_areas["EXOGENOUS"] = exogenous_busbars
config.market_calibration_areas += config.market_calibration_areas.from_dict(_market_calibration_areas)

# %%  Lines


def loss_percentage_fn(capacity: float, resistance: float, voltage: float = 400) -> float:
    """Base voltage of PowerGama is 400 kV"""
    power = capacity * 1e6  # MW
    current = (power) / (np.sqrt(3) * voltage * 1e3)
    p_loss = current**2 * resistance
    loss_percentage = p_loss / power
    if p_loss < 1e-6:  # Not available for DC Lines
        loss_percentage = 0.03
    return loss_percentage


link_pairs = defaultdict(lambda: defaultdict(dict))
for _, row in data_loader.branch.iterrows():
    pair = tuple(sorted([row["node_from"], row["node_to"]]))
    direction = (row["node_from"], row["node_to"])
    link_pairs[pair][direction]["capacity"] = row["capacity"]
    link_pairs[pair][direction]["loss_percentage"] = loss_percentage_fn(
        row["capacity"], resistance=row["resistance_ohm"]
    )

for link in link_pairs:
    a2b_capacity = link_pairs[link].get((link[0], link[1]), {}).get("capacity")
    b2a_capacity = link_pairs[link].get((link[1], link[0]), {}).get("capacity", a2b_capacity)

    if a2b_capacity is None:
        a2b_capacity = b2a_capacity

    a2b_loss_percentage = link_pairs[link].get((link[0], link[1]), {}).get("loss_percentage")
    b2a_loss_percentage = link_pairs[link].get((link[1], link[0]), {}).get("loss_percentage", a2b_loss_percentage)

    if a2b_loss_percentage is None:
        a2b_loss_percentage = b2a_loss_percentage

    loss_percentage = min(max(a2b_loss_percentage, b2a_loss_percentage), 0.05)

    line = DcLine(
        name=f"{link[0]} {link[1]}",
        backward_capacity=TimeSeries(config=ConstantTimeseriesConfig(value=a2b_capacity), unit=Unit.MW),
        forward_capacity=TimeSeries(config=ConstantTimeseriesConfig(value=b2a_capacity), unit=Unit.MW),
        loss_percentage=loss_percentage,
    )

    _logger.info(
        f"Adding dcline {line.name} with capacity {a2b_capacity}/{b2a_capacity} MW and loss percentage {loss_percentage}"
    )

    config.add(line)

    config.connect(to_obj=line, from_obj=busbars_obj[link[0]])
    config.connect(to_obj=busbars_obj[link[1]], from_obj=line)

# %% Load

simulation_time_index = pd.date_range(start=simulation_start, end=simulation_end, freq="1h")
dfs_load = {}

if load_per_scenario:
    for idx, row in data_loader.consumer.iterrows():
        node = row["node"]

        if node in ["EE", "FI", "LT"]:  # EE & LT Not available in load profiles, assuming similar profile as
            node = "FIN"
        elif node == "GB":
            node = "GBR"
        elif node == "DE":
            node = "DEU"
        elif node == "NL":
            node = "NLD"
        elif node == "PL":
            node = "DEU"  # Approximate Poland with Germany

        df_load = pd.read_csv(f"data/Profiler/Consumption/{node}_consumption.csv", index_col=0, parse_dates=True)
        df_load = df_load.loc[(df_load.index.year >= start_scenario_year) & (df_load.index.year <= end_scenario_year)]

        data = []
        for scenario in range(start_scenario_year, end_scenario_year + 1):
            ind = df_load.index.year >= scenario
            end_ind = len(simulation_time_index)

            data.append(
                np.squeeze(df_load[ind][:end_ind].values)
            )  # TODO: Handle to start from start if more than n_years

        data = np.array(data)
        df = pd.DataFrame(
            index=simulation_time_index, data=data.T, columns=range(start_scenario_year, end_scenario_year + 1)
        )

        dff = df / df.mean()  # Normalize to average 1.0
        dff = dff * row["demand_avg"]
        dfs_load[row["node"]] = dff

        # if row["node"] in ["DK1", "DK2"]:  # NB! WARNING!
        #     scaling_factor = 1.6
        #     dff = dff * scaling_factor
        #     _logger.warning(f"Scaling load for {row['node']} with {scaling_factor} to test negative residual load error")

        consumption = Load(
            name=row["Load"],
            capacity=TimeSeries(value=dff.copy(), unit=Unit.MW, enforce_scenario_dimensions=True),
        )
        config.add(consumption)
        config.connect(to_obj=busbars_obj[row["node"]], from_obj=consumption)

        _logger.info(f"Adding Load for {row['node']}, avg {row['demand_avg']:.2f} MW")

        # (row["demand_avg"]*dff).plot(title=f"Load profile for {row['node']}", figsize=(12, 6))


else:
    for idx, row in data_loader.consumer.iterrows():
        df_load = pd.DataFrame(
            data=df_load_profiles[row["demand_ref"]].values[: len(simulation_time_index)], index=simulation_time_index
        )

        dff = df_load * row["demand_avg"]
        dfs_load[row["node"]] = dff

        consumption = Load(
            name=row["Load"],
            capacity=TimeSeries(value=dff.copy(), unit=Unit.MW, enforce_scenario_dimensions=True),
        )
        config.add(consumption)
        config.connect(to_obj=busbars_obj[row["node"]], from_obj=consumption)

        _logger.info(f"Adding Load for {row['node']}, avg {row['demand_avg']:.2f} MW")


# %% Generators


for idx, row in data_loader.generator.iterrows():
    desc = row["desc"]

    if row["type"] in (["wind_off", "wind_on", "solar"]):
        if row["type"] in (["wind_off", "wind_on"]):
            obj_type = Wind
        else:
            obj_type = Solar

        inflow_ref = row["inflow_ref"]
        df_temp = (df_renewables_profiles.loc[:, inflow_ref] * row["pmax"]).to_frame()

        data = []
        for scenario in range(start_scenario_year, end_scenario_year + 1):
            ind = df_temp.index.year >= scenario
            end_ind = len(simulation_time_index)

            data.append(
                np.squeeze(df_temp[ind][:end_ind].values)
            )  # TODO: Handle to start from start if more than n_years

        data = np.array(data)
        df = pd.DataFrame(
            index=simulation_time_index, data=data.T, columns=range(start_scenario_year, end_scenario_year + 1)
        )

        market_step = obj_type(
            name=desc,
            capacity=TimeSeries(
                value=df.copy(),
                unit=Unit.MW,
                enforce_scenario_dimensions=True,
            ),
        )

        _logger.info(f"Adding {obj_type} for {row['node']} and type {row['type']}")
        config.add(market_step)
        config.connect(to_obj=busbars_obj[row["node"]], from_obj=market_step)

    elif row["type"] in (["biomass", "fossil_gas", "fossil_other", "nuclear"]):
        if row["type"] == "nuclear":
            if IMPROVE_NUCLEAR_REP:
                add_historic_nuclear(
                    config=config,
                    node=row["node"],
                    capacity=row["pmax"],
                    logger=_logger,
                )
                _logger.info(f"Adding firm historic nuclear for {row['node']} as Wind object")
            else:
                add_historic_flexible_nuclear(
                    config=config,
                    node=row["node"],
                    capacity=row["pmax"],
                    price=row.get("fuelcost", NEW_NUCLEAR_MARGINAL_COST),
                    logger=_logger,
                )
                _logger.info(
                    f"Adding price-dependent historic nuclear for {row['node']} using new nuclear profile"
                )
            continue

        else:
            market_step = MarketStep(
                name=desc,
                capacity=TimeSeries(config=ConstantTimeseriesConfig(value=row["pmax"]), unit=Unit.MW),
                price=TimeSeries(config=ConstantTimeseriesConfig(value=row["fuelcost"]), unit=Unit.EUR_MWH),
                fuel_type=row["type"],
            )

            _logger.info(f"Adding generator for {row['node']} and type {row['type']}")
        config.add(market_step)
        config.connect(to_obj=busbars_obj[row["node"]], from_obj=market_step)

    # elif row["type"] in (["hydro"]):
    #     if row["node"][:2] not in (["NO", "SE"]):  # Detailed hydro in Norway and Sweden from NVE #TODO: Handle regulated vs unregualted inflow to Finland
    #         df_temp = (df_renewables_profiles[row["inflow_ref"]] * row["pmax"] * row["inflow_fac"]).to_frame()

    #         data = []
    #         for scenario in range(start_scenario_year, end_scenario_year + 1):
    #             ind = df_temp.index.year >= scenario
    #             end_ind = len(simulation_time_index)

    #             data.append(
    #                 np.squeeze(df_temp[ind][:end_ind].values)
    #             )  # TODO: Handle to start from start if more than n_years

    #         data = np.array(data)
    #         df = pd.DataFrame(index=simulation_time_index, data=data.T)

    #         hydro_power = AggregatedHydroModule(
    #             name=row["node"],
    #             regulated_energy_inflow=Timeseries(
    #                 value=df,
    #                 unit=Unit.MW,
    #                 enforce_scenario_dimensions=True,
    #             ),
    #             unregulated_energy_inflow=Timeseries(
    #                 config=ConstantTimeseriesConfig(value=0), unit=Unit.MW, enforce_scenario_dimensions=True
    #             ),
    #             reservoir_energy=row["storage_cap"] ,  # NB! Assume it is in MWh
    #             station_power=row["pmax"], # Assume it is in MW
    #             lower_reservoir_limits=Timeseries(
    #                 config=ConstantTimeseriesConfig(value=0),
    #                 unit=Unit.MW,
    #             ),
    #             lower_production_limits=Timeseries(
    #                 config=ConstantTimeseriesConfig(value=row["pmin"]),
    #                 unit=Unit.MW,
    #             ),
    #             upper_production_limits=Timeseries(
    #                 config=ConstantTimeseriesConfig(value=row["pmax"]),
    #                 unit=Unit.MW,
    #             ),
    #             upper_reservoir_limits=Timeseries(
    #                 config=ConstantTimeseriesConfig(value=row["storage_cap"] / 1e3),
    #                 unit=Unit.MW,
    #             ),
    #             start_reservoir_energy=row["storage_cap"] * start_reservoir_value_pu, # Should be MWh
    #         )

    #         _logger.info(f"Adding AggregatedHydroModule for {row['node']}")
    #         config.add(hydro_power)
    #         config.connect(to_obj=busbars_obj[row["node"]], from_obj=hydro_power)

    else:  # hydro is defined by NVE dataset
        continue


# %% Add NVE data
_logger.info("Extending model from NVE")
config.extend_from_nve(use_most_common_elspot=True)

# %% Add inflow series for Finland

# Combining two northern inflow series with one southern to represtent the Finnish
# Using: 234.18 (polmak, 2960-E) and 247.3 (karpelv, 2614-E) with 2.142 (knapphom, 410-E)
# See: https://publikasjoner.nve.no/rapport/2008/rapport2008_07.pdf

inflow_path = files("lpr_sintef_bifrost.inputs.files.nve") / "inflow.h5"


def get_inflow_series_as_df(inflow_code: str):
    with h5py.File(Path(inflow_path), "r") as h5file:
        data = h5file[f"/{inflow_code}/vals"][:]
        index = h5file[f"/{inflow_code}/times"][:]

    return pd.DataFrame(
        data,
        index=pd.to_datetime(index, unit="ms").tz_localize(CET_winter),
        columns=[inflow_code],
    )


df = pd.concat(
    [
        get_inflow_series_as_df("2960-E"),
        get_inflow_series_as_df("2614-E"),
        get_inflow_series_as_df("410-E"),
    ],
    axis=1,
)

df = df[df > 0.0]
inf_avg = df.mean().mean()
df["9998-N"] = inf_avg * (df / df.mean()).mean(axis=1).fillna(-9999)

inflow_series_finland = InflowSeries(
    name="9998-N",
    metadata=ModelMetadata(series_name="approx-finland"),
    series=TimeSeries(value=df[["9998-N"]], name="9998-N", unit=Unit.m3s),
)

_logger.info("Adding InflowSeries 9998-N")
config.inflow_series.add_item(inflow_series_finland)

fi_reservoir = config.reservoirs.get("finland")
fi_reservoir.metadata.reservoir_capacity_mm3 = 10  # Dummy value to avoid issues with water values

# %% Exogenouse prices

if use_exogenous_prices:
    # Add exogenous market steps
    _logger.info(f"Adding exogenous market steps to model for busbars: {exogenous_busbars}")

    if len(config.price_series_main) == 0:
        # This might not be needed at all (?)
        _logger.info("No price series found in the model. Adding default price series.")
        config.price_series_main.add_item(
            PriceSeries(
                name="default_price_series",
                series=TimeSeries(
                    config=ConstantTimeseriesConfig(value=40),
                    enforce_scenario_dimensions=True,
                    unit=Unit.EUR_MWH,
                ),
            )
        )
    for exogenous_busbar in exogenous_busbars:
        prices_path = {
            "DE": "data/de_price.parquet",
            "PL": "data/pl_price.parquet",
            "GB": "data/uk_price.parquet",
            "NL": "data/nl_price.parquet",
        }

        df = pd.read_parquet(prices_path[exogenous_busbar])

        price_series = PriceSeries(
            name=f"{exogenous_busbar}_PRICE_SERIES",
            series=TimeSeries(
                value=df,
                enforce_scenario_dimensions=True,
                unit=Unit.EUR_MWH,
            ),
        )
        config.price_series_secondary.add_item(price_series)

        market_step_sell = MarketStep(
            name=f"{exogenous_busbar}_sell",
            capacity=TimeSeries(
                config=ConstantTimeseriesConfig(value=1_000_000),
                unit=Unit.MW,
            ),
            secondary_price_series_name=price_series.name,
        )
        market_step_buy = MarketStep(
            name=f"{exogenous_busbar}_buy",
            capacity=TimeSeries(
                config=ConstantTimeseriesConfig(value=-1_000_000),
                unit=Unit.MW,
            ),
            secondary_price_series_name=price_series.name,
        )

        config.add(market_step_sell)
        config.add(market_step_buy)
        busbar = config.busbars.get(exogenous_busbar)
        config.connect(market_step_sell, busbar)
        config.connect(market_step_buy, busbar)


# %%  Filter the collections
_logger.info(f"Filtering model with busbars: {busbars}, exogenous busbars: {exogenous_busbars}")
config.filter(ModelFilter(busbars=busbars, exogenous_busbars=exogenous_busbars))

# Filter the aggregated hydropower modules
_logger.info(f"Keep aggregated hydropower modules with busbars: {aggregated_hydropower_busbars}")
config.aggregated_hydro_modules.filter(model_filter=ModelFilter(busbars=aggregated_hydropower_busbars))

# %% Set start-reservoir

for module in config.aggregated_hydro_modules:
    _logger.info(f"Setting start reservoir level for {module.name} to {start_reservoir_value_pu * 100:.2f}%")
    module.start_reservoir_energy = module.reservoir_energy * start_reservoir_value_pu

_logger.info(f"Setting start reservoir levels to {start_reservoir_value_pu * 100:.2f}%")
for reservoir in config.reservoirs:
    reservoir.initial_volume = reservoir.metadata.reservoir_capacity_mm3 * start_reservoir_value_pu

for plant in config.plants:
    plant.unavailable_capacity = None

for pump in config.pumps:
    pump.unavailable_capacity = None

# %% Avoid issue with min and max volume curces for Byglandsfjord

config.reservoirs["byglandsfjord"].min_volume_curve = None
config.reservoirs["byglandsfjord"].max_volume_curve = None

# %% Allow negative residual load

if False:
    _logger.warning("Allowing validation failures")
    config.global_settings.allow_validation_failures = True

# %% Save model

config.global_settings.num_processes_override = None

_logger.info("Writing dataset to json-file.")
config.to_json(filepath=destination_folder / "dataset.json")

# %%

run_config = {
    "dataset_year": dataset_year,
    "dataset_scenario": dataset_scenario,
    "resolution": resolution,
    "simulation_type": simulation_type,
    "use_exogenous_prices": use_exogenous_prices,
    "improve_nuclear_rep": IMPROVE_NUCLEAR_REP,
    "base_nuclear_profile": HISTORIC_NUCLEAR_PROFILE_PATH if IMPROVE_NUCLEAR_REP else NEW_NUCLEAR_PROFILE_PATH,
    "base_nuclear_price_dependent": not IMPROVE_NUCLEAR_REP,
    "historic_nuclear_profile": HISTORIC_NUCLEAR_PROFILE_PATH,
    "new_nuclear_profile": NEW_NUCLEAR_PROFILE_PATH,
    "new_nuclear_firm_share": NEW_NUCLEAR_FIRM_SHARE if IMPROVE_NUCLEAR_REP else 0.0,
    "new_nuclear_flexible_share": 1.0 - (NEW_NUCLEAR_FIRM_SHARE if IMPROVE_NUCLEAR_REP else 0.0),
    "new_nuclear_marginal_cost": NEW_NUCLEAR_MARGINAL_COST,
    "nuclear_market_step_capacity_scenario_independent": True,
}

_logger.info("Writing run config file.")
with open(destination_folder / "config.txt", "w") as f:
    for key, value in run_config.items():
        f.write(f"{key}: {value}\n")
