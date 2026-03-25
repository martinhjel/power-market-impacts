# %%
import io
import json
import logging
import uuid
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

# %%
import numpy as np
import pandas as pd
from lpr_sintef_bifrost.inputs import TimeSeries
from lpr_sintef_bifrost.inputs.timeseries import ConstantTimeseriesConfig
from lpr_sintef_bifrost.models import EMPSModelBuilder
from lpr_sintef_bifrost.models.common import (
    Busbar,
    Load,
    MarketStep,
)
from lpr_sintef_bifrost.models.connection import ObjectType
from lpr_sintef_bifrost.models.emps import DcLine, Wind
from lpr_sintef_bifrost.models.emps._feedback_factors import (
    FeedbackConversionLimit,
    FeedbackFactor,
    FeedbackFactors,
)
from lpr_sintef_bifrost.utils.unit import Unit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger("lpr_sintef_bifrost")

# %%


def add_area_with_offshore_wind(
    config,
    new_area_name: str,
    connected_area: str,
    transmission_capacity_mw: float,
    offshore_profile_name: str,
    offshore_pmax: float,
    df_renewables_profiles: pd.DataFrame,
    logger=None,
):
    # Create logger if not provided
    if logger is None:
        logger = logging.getLogger("lpr_sintef_bifrost")

    # 1. Add new Busbar
    new_busbar = Busbar(name=new_area_name)
    config.add(new_busbar)
    logger.info(f"Added new busbar: {new_area_name}")

    busbars_obj = {}
    for b in config.busbars:
        busbars_obj[b.name] = b

    # 2. Ensure market calibration area includes the new busbar
    target_market_area_name = None
    for connection in config.market_calibration_areas.connections:
        if (
            connection.from_name == connected_area
            and connection.to_type == ObjectType.MARKET_CALIBRATION_AREA
        ):
            target_market_area_name = connection.to_name
            break

    if target_market_area_name is not None:
        target_market_area = config.market_calibration_areas.get(target_market_area_name)
        existing_link = any(
            conn.from_name == new_area_name and conn.to_name == target_market_area_name
            for conn in config.market_calibration_areas.connections
        )
        if not existing_link:
            config.market_calibration_areas.add_connection(
                from_obj=new_busbar, to_obj=target_market_area
            )
            logger.info(
                f"Linked {new_area_name} to market calibration area {target_market_area_name}"
            )
    else:
        logger.warning(
            f"Could not determine market calibration area for {connected_area}; "
            f"{new_area_name} not added to market calibration areas."
        )

    # 3. Create DC Line from existing area
    line_name = f"{connected_area} {new_area_name}"
    line = DcLine(
        name=line_name,
        backward_capacity=TimeSeries(config=ConstantTimeseriesConfig(value=transmission_capacity_mw), unit=Unit.MW),
        forward_capacity=TimeSeries(config=ConstantTimeseriesConfig(value=transmission_capacity_mw), unit=Unit.MW),
        loss_percentage=0.03,  # Simplified default; can also compute based on resistance if needed
    )
    config.add(line)
    config.connect(to_obj=line, from_obj=busbars_obj[connected_area])
    config.connect(to_obj=new_busbar, from_obj=line)
    logger.info(f"Connected {new_area_name} to {connected_area} with {transmission_capacity_mw} MW DC line")

    # 4. Offshore Wind: Generate time series with scenario dimension
    if offshore_profile_name not in df_renewables_profiles.columns:
        raise ValueError(f"Profile '{offshore_profile_name}' not found in df_renewables_profiles")

    profile = df_renewables_profiles[offshore_profile_name] * offshore_pmax
    profile.name = new_area_name + "_offshore_wind"
    data = []

    simulation_time_index = pd.date_range(start=config.start, end=config.end, freq="1h")

    for scenario in range(config.start_scenario_year, config.end_scenario_year + 1):
        ind = profile.index.year >= scenario
        end_ind = len(simulation_time_index)
        data.append(profile[ind][:end_ind].values)

    df_capacity = pd.DataFrame(index=simulation_time_index, data=np.array(data).T)

    wind = Wind(
        name=new_area_name + "_offshore_wind",
        capacity=TimeSeries(
            value=df_capacity,
            unit=Unit.MW,
            enforce_scenario_dimensions=True,
        ),
    )

    config.add(wind)
    config.connect(to_obj=busbars_obj[new_area_name], from_obj=wind)
    logger.info(
        f"Added offshore wind generator to {new_area_name} with profile '{offshore_profile_name}' and Pmax={offshore_pmax} MW"
    )


def get_busbar(config, node: str):
    for busbar in config.busbars:
        if busbar.name == node:
            return busbar


def add_nuclear(config, node: str, capacity: float, price: float, capacity_factor: float = 0.90):
    
    df_nuclear_profile = pd.read_parquet("data/new_nuclear_profile.parquet")
        
    market_step = MarketStep(
        name=f"{node} nuclear {uuid.uuid4().hex[:2]}",
        capacity=TimeSeries(value=df_nuclear_profile*capacity, unit=Unit.MW),
        price=TimeSeries(config=ConstantTimeseriesConfig(value=price), unit=Unit.EUR_MWH),
        fuel_type="nuclear",
    )

    _logger.info(f"Adding MarketStep for {node} and type nuclear. {capacity} MW and marginal cost {price} EUR/MWh.")
    config.add(market_step)
    config.connect(to_obj=get_busbar(config, node), from_obj=market_step)


def add_baseload(config, node: str, capacity: float):
    consumption = Load(
        name=f"Load {uuid.uuid4().hex[:4]}",
        capacity=TimeSeries(
            config=ConstantTimeseriesConfig(value=capacity), unit=Unit.MW, enforce_scenario_dimensions=True
        ),
    )
    _logger.info(f"Adding {capacity} MW baseload for {node}.")
    config.add(consumption)
    config.connect(to_obj=get_busbar(config, node), from_obj=consumption)


def get_uprated_capacity_by_area(config, uprate_values):
    area_added_capacity = defaultdict(float)

    for plant in uprate_values:
        plant_obj = [i for i in config.plants.find(plant) if i.name == plant][0]

        if plant_obj:
            if len(plant_obj.pq_curves.config.x) == 2:
                added_capacity = plant_obj.pq_curves.config.y[-1] * (uprate_values[plant_obj.name]["uprate"] - 1)
                area = uprate_values[plant]["elspot_area"]
                area_added_capacity[area] += added_capacity

    return dict(area_added_capacity)


def uprate_hydropower(config, uprate_values: dict):
    accumulated_added_capacity = 0.0
    for plant in uprate_values:
        plant_obj = [i for i in config.plants.find(plant) if i.name == plant][0]

        if plant_obj:
            if isinstance(plant_obj.discharge_energy_equivalent.config, ConstantTimeseriesConfig):
                _logger.info(
                    f"Improving energy equivalent with {(uprate_values[plant_obj.name]['eff'] - 1) * 100:.2f}% for {plant_obj.name}."
                )
                plant_obj.discharge_energy_equivalent.config.value *= uprate_values[plant_obj.name]["eff"]

            if len(plant_obj.pq_curves.config.x) == 2:
                added_capacity = plant_obj.pq_curves.config.y[-1] * (uprate_values[plant_obj.name]["uprate"] - 1)
                _logger.info(
                    f"Adding {added_capacity:.2f} MW to {plant_obj.name}. Existing is {plant_obj.pq_curves.config.y[-1]:.2f} MW"
                )

                accumulated_added_capacity += added_capacity
                plant_obj.pq_curves.config.x[-1] *= uprate_values[plant_obj.name]["uprate"]
                plant_obj.pq_curves.config.y[-1] *= uprate_values[plant_obj.name]["uprate"]

            if plant_obj.max_discharge_curve:
                date_vals = plant_obj.max_discharge_curve.config.date_values
                date_vals = [(i, j * uprate_values[plant_obj.name]["uprate"]) for i, j in date_vals]
                _logger.info(f"Setting max_discharge_curve for {plant_obj.name} to {date_vals}")
                plant_obj.max_discharge_curve.config.date_values = date_vals

            if plant_obj.min_discharge_curve:
                date_vals = plant_obj.min_discharge_curve.config.date_values
                date_vals = [(i, j * uprate_values[plant_obj.name]["uprate"]) for i, j in date_vals]
                _logger.info(f"Setting min_discharge_curve for {plant_obj.name} to {date_vals}")
                plant_obj.min_discharge_curve.config.date_values = date_vals

            for res in uprate_values[plant]["reservoirs"]:
                res_obj = [i for i in config.reservoirs.find(res) if i.name == res][0]
                max_discharge = res_obj.max_discharge * uprate_values[plant_obj.name]["uprate"]
                _logger.info(f"Setting max_discharge for reservoir {res_obj.name} to {max_discharge}")
                res_obj.max_discharge = max_discharge

                _logger.info(
                    f"The area for plant {plant_obj.name} and {res_obj.name} is {res_obj.metadata.origin_area}"
                )

    _logger.info(f"Accumulated added hydro power capacity is {accumulated_added_capacity:.2f} MW")


def _ensure_dataframe(value, load_name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    raise ValueError(f"Load '{load_name}' does not contain inline timeseries data.")


def _get_time_step_hours(index: pd.Index) -> float:
    if len(index) < 2:
        return 1.0
    delta = index[1] - index[0]
    return delta.total_seconds() / 3600.0


def _get_norwegian_loads(config):
    norway_prefixes = {f"NO{i}" for i in range(1, 6)}
    loads = [load for load in config.loads if load.name in norway_prefixes]
    return loads


def add_load_as_ba(config, new_yearly_load_twh: float):
    if new_yearly_load_twh <= 0:
        _logger.info("No baseload addition requested.")
        return

    loads = _get_norwegian_loads(config)
    if not loads:
        _logger.warning("No Norwegian loads found; skipping baseload addition.")
        return

    load_dfs = {}
    load_means = {}
    for load in loads:
        df = _ensure_dataframe(load.capacity.value, load.name).copy()
        load_dfs[load.name] = df
        load_means[load.name] = df.mean().mean()

    total_mean = sum(load_means.values())
    if total_mean <= 0:
        _logger.warning("Total Norwegian load mean is zero; skipping baseload addition.")
        return

    template_df = next(iter(load_dfs.values()))
    timestep_hours = _get_time_step_hours(template_df.index)
    total_hours = len(template_df.index) * timestep_hours
    if total_hours <= 0:
        _logger.warning("Could not determine total hours for baseload addition; skipping.")
        return

    avg_new_demand_mw = (new_yearly_load_twh * 1e6) / total_hours
    _logger.info(
        f"Adding {new_yearly_load_twh:.2f} TWh baseload across Norwegian areas "
        f"({avg_new_demand_mw:.2f} MW average)."
    )

    for load in loads:
        share = load_means[load.name] / total_mean
        addition_mw = share * avg_new_demand_mw
        if addition_mw <= 0:
            continue

        df_increment = pd.DataFrame(
            addition_mw, index=template_df.index, columns=template_df.columns
        )
        consumption = Load(
            name=f"Baseload_{load.name}_{int(round(addition_mw))}MW",
            capacity=TimeSeries(
                value=df_increment,
                unit=Unit.MW,
                enforce_scenario_dimensions=True,
            ),
        )
        _logger.info(
            f"Adding {addition_mw:.2f} MW baseload for {load.name} ({share:.2%} share)."
        )
        config.add(consumption)
        config.connect(to_obj=get_busbar(config, load.name), from_obj=consumption)


def add_load_as_llps(config, new_yearly_load_twh: float):
    if new_yearly_load_twh <= 0:
        _logger.info("No LLPS addition requested.")
        return

    loads = _get_norwegian_loads(config)
    if not loads:
        _logger.warning("No Norwegian loads found; skipping LLPS scaling.")
        return

    load_dfs = {}
    load_means = {}
    for load in loads:
        df = _ensure_dataframe(load.capacity.value, load.name).copy()
        load_dfs[load.name] = df
        load_means[load.name] = df.mean().mean()

    total_mean = sum(load_means.values())
    if total_mean <= 0:
        _logger.warning("Total Norwegian load mean is zero; skipping LLPS scaling.")
        return

    template_df = next(iter(load_dfs.values()))
    timestep_hours = _get_time_step_hours(template_df.index)
    total_hours = len(template_df.index) * timestep_hours
    if total_hours <= 0:
        _logger.warning("Could not determine total hours for LLPS scaling; skipping.")
        return

    avg_new_demand_mw = (new_yearly_load_twh * 1e6) / total_hours
    _logger.info(
        f"Applying LLPS to add {new_yearly_load_twh:.2f} TWh across Norwegian areas "
        f"({avg_new_demand_mw:.2f} MW average)."
    )

    for load in loads:
        base_mean = load_means[load.name]
        if base_mean <= 0:
            _logger.warning(f"Load {load.name} has zero mean; skipping LLPS scaling.")
            continue

        share = base_mean / total_mean
        additional_avg_mw = share * avg_new_demand_mw
        scale_factor = 1.0 + additional_avg_mw / base_mean
        df_scaled = load_dfs[load.name] * scale_factor

        _logger.info(
            f"Scaling load {load.name} by factor {scale_factor:.4f} "
            f"({additional_avg_mw:.2f} MW additional, {share:.2%} share)."
        )

        load.capacity = TimeSeries(
            value=df_scaled,
            unit=Unit.MW,
            enforce_scenario_dimensions=True,
        )


# %%


def update_feedback_factors(config):
    with open(Path.cwd() / "data/feedback_factors.json", "r") as f:
        calibration_factors = json.load(f)

    for busbar in config.busbars:
        factors = calibration_factors[busbar.name]
        _logger.info(f"Setting feedback factors for {busbar.name} to {factors}")
        busbar.feedback_factors = FeedbackFactors(
            artificial_minimum_production=1.0,
            conversion_limit=FeedbackConversionLimit(
                value=factors["cl"],
                normal_value=factors["cl"],  # now using conversion limit
            ),
            feedback_factor=FeedbackFactor(
                normal_value=factors["feedf"],
                start_week_values=TimeSeries(
                    config=ConstantTimeseriesConfig(value=factors["feedf"]),
                    unit=Unit.unitless,
                ),
            ),
            form_factor=FeedbackFactor(
                normal_value=factors["formf"],
                start_week_values=TimeSeries(
                    config=ConstantTimeseriesConfig(
                        value=factors["formf"],
                    ),
                    unit=Unit.unitless,
                ),
            ),
            flexibility_factor=FeedbackFactor(
                normal_value=factors["flexf"],
                start_week_values=TimeSeries(
                    config=ConstantTimeseriesConfig(
                        value=factors["flexf"],
                    ),
                    unit=Unit.unitless,
                ),
            ),
        )


def format_reservoir(res):
    def format_ts(ts):
        if ts is None or ts.config is None:
            return "N/A"
        if hasattr(ts.config, "date_values"):
            return "\n        " + "\n        ".join(f"• {d}: {v}" for d, v in ts.config.date_values)
        elif hasattr(ts.config, "value"):
            return f"{ts.config.value} (constant)"
        return str(ts.config)

    def format_volume_curve(vc):
        if vc is None or vc.config is None:
            return "N/A"
        x = vc.config.x
        y = vc.config.y
        return "\n        " + "\n        ".join(f"{xi} m → {yi} Mm³" for xi, yi in zip(x, y))

    meta = res.metadata
    print(f"Reservoir: {meta.name}")
    print(f"├── Watercourse: {meta.watercourse}")
    print(f"├── Origin Area: {meta.origin_area}")
    print(f"├── Module ID: {meta.module_id}")
    print(f"├── Capacity: {meta.reservoir_capacity_mm3:.2f} Mm³")
    print(f"├── Initial Volume: {res.initial_volume:.2f} Mm³")
    print(f"├── Degree of Regulation: {res.degree_of_regulation}")
    print(f"├── Gross Head: {res.gross_head} m")
    print(f"├── Tailrace Elevation: {res.tailrace_elevation} m")
    print(f"├── Max Discharge: {res.max_discharge} m³/s")
    print(f"├── Avg. Spill Energy Equivalent: {res.average_spill_energy_equivalent}")
    print(f"├── Avg. Regulated Inflow: {res.average_regulated_inflow:.2f} Mm³/week")
    print(f"├── Regulated Inflow Name: {res.regulated_inflow_name}")
    print(f"├── Reference Curve:{format_ts(res.reference_curve)}")
    print(f"├── Volume Curve (Elevation vs Volume):{format_volume_curve(res.volume_curve)}")
    print(f"└── Max Bypass Flow: {format_ts(res.max_bypass_curve)}")


def format_plant(plant):
    def format_ts(ts):
        if ts is None or ts.config is None:
            return "N/A"
        if hasattr(ts.config, "value"):
            return f"{ts.config.value} (constant)"
        return str(ts.config)

    def format_xy_curve(curve):
        if curve is None or curve.config is None:
            return "N/A"
        x = curve.config.x
        y = curve.config.y
        return "\n        " + "\n        ".join(
            f"{xi} {curve.x_unit.value if curve.x_unit else ''} → {yi} {curve.y_unit.value if curve.y_unit else ''}"
            for xi, yi in zip(x, y)
        )

    meta = plant.metadata

    print(f"Plant: {meta.name}")
    print(f"├── Watercourse: {meta.watercourse}")
    print(f"├── NVE Name: {meta.nve_name}")
    print(f"├── Gross Head: {plant.gross_head} m")
    print(f"├── Tailrace Elevation: {plant.tailrace_elevation} m")
    print(f"├── Ownership: {plant.ownership}%")
    print(f"├── Avg. Unregulated Inflow: {plant.average_unregulated_inflow:.2f} m³/s")
    print(f"├── Unregulated Inflow Name: {plant.unregulated_inflow_name}")
    print(f"├── Discharge Energy Equivalent: {format_ts(plant.discharge_energy_equivalent)}")
    print(f"├── PQ Curve (Discharge → Power):{format_xy_curve(plant.pq_curves)}")
    print(f"└── Max Discharge Curve: {'Defined' if plant.max_discharge_curve else 'None'}")


def format_pump(pump):
    def format_ts(ts):
        if ts is None or ts.config is None:
            return "N/A"
        if hasattr(ts.config, "value"):
            return f"{ts.config.value} (constant)"
        return str(ts.config)

    def format_xy_curve(curve):
        if curve is None or curve.config is None:
            return "N/A"
        if hasattr(curve.config, "x") and hasattr(curve.config, "y"):
            x = curve.config.x
            y = curve.config.y
        elif hasattr(curve, "value") and hasattr(curve.value, "items"):
            # Try to extract from DataFrame-like object
            try:
                x = list(curve.value.index)
                y = list(curve.value.iloc[:, 0])
            except Exception:
                return "Malformed data"
        else:
            return "N/A"

        return "\n        " + "\n        ".join(
            f"{xi} {curve.x_unit.value if curve.x_unit else ''} → {yi} {curve.y_unit.value if curve.y_unit else ''}"
            for xi, yi in zip(x, y)
        )

    meta = pump.metadata

    print(f"Pump: {pump.name}")
    print(f"├── Watercourse: {meta.watercourse}")
    print(f"├── Ownership: {pump.ownership}%")
    print(f"├── Average Power: {pump.average_power} MW")
    print(f"├── Pump Capacity Curve (Head → Flow):{format_xy_curve(pump.pump_capacity)}")
    print(f"├── Upper Reservoir Ref. Level: {format_ts(pump.upper_reservoir_reference_curve)}")
    print(f"├── Lower Reservoir Ref. Level: {format_ts(pump.lower_reservoir_reference_curve)}")
    print(f"└── Unavailable Capacity: {'Defined' if pump.unavailable_capacity else 'None'}")


def find_connections_all(config, connection_search: str):
    connections = []
    for c in config.connections:
        if c.from_name == connection_search or c.to_name == connection_search:
            connections.append(c)

    return connections


def find_connections_upper(config, connection_search: str):
    connections = []
    for c in config.connections:
        if c.to_name == connection_search:
            connections.append(c)

    return connections


def main():
    # %%
    model_name = "PowerGamaMSc"
    use_exogenous_prices = True
    resolution = "1D"
    simulation_type = "serial"  # ["parallel", "serial"]
    dataset_year = 2025
    dataset_scenario = "BM"

    destination_folder = (
        Path().cwd()
        / f"ltm_output/{model_name}_{dataset_year}_{dataset_scenario}_{resolution}_{simulation_type}_{use_exogenous_prices}EXO_detFi_FF/"
    )

    dataset_path = destination_folder / "dataset.json"
    config = EMPSModelBuilder.from_json(filepath=dataset_path)

    busbars_obj = {}
    for busbar in config.busbars:
        busbars_obj[busbar.name] = busbar

    add_nuclear(config, node="NO2", capacity=900, price=10)
    add_baseload(config=config, node="NO2", capacity=500)

    # %%
    uprate_values = {
        "nore_i": {
            "eff": 1700 / 1500,
            "uprate": 2,
            "reservoirs": ["nore_1"],
        },  # Source: https://constructionreviewonline.com/news/statkraft-submits-plan-to-upgrade-nore-power-plant-in-norway-for-4-billion-nok/?utm_source=chatgpt.com
        "nore_ii": {
            "eff": 1700 / 1500,
            "uprate": 2,
            "reservoirs": ["nore_2"],
        },  # Source: https://constructionreviewonline.com/news/statkraft-submits-plan-to-upgrade-nore-power-plant-in-norway-for-4-billion-nok/?utm_source=chatgpt.com
        "mauranger": {
            "eff": (1150 + 75) / 1150,
            "uprate": 880 / 250,  # Source: https://no.wikipedia.org/wiki/Mauranger_kraftverk
            "reservoirs": ["mauranger"],
        },
        "aura": {
            "eff": 1.05,  # Assumed, 290 MW installed
            "uprate": 810 / 310,  # Source: https://energiwatch.no/nyheter/fornybar/article18204332.ece
            "reservoirs": ["aura"],
        },
        "osbu": {
            "eff": 1.05,  # Assumed, 20 MW installed
            "uprate": 810 / 310,  # Source: https://energiwatch.no/nyheter/fornybar/article18204332.ece
            "reservoirs": ["osbu"],
        },
        "alta": {
            "eff": ((100 + 150) / 2 + 694.7) / 694.7,  # 150 MW installed
            "uprate": (120 + 150)
            / 150,  # Source: https://www.nve.no/media/17040/alta-kraftverk-a3-prosjektbeskrivelse-med-utredningsprogram.pdf
            "reservoirs": ["alta"],
        },
        "svean": {
            "eff": (10 + 129.8) / 129.8,
            "uprate": 36
            / 27,  # Source: https://www.statkraft.no/om-statkraft/hvor-vi-har-virksomhet/norge/svean-vannkraftverk/, https://www.statkraft.no/presserom/nyheter-og-pressemeldinger/2025/statkraft-signerer-avtaler-for-kraftverk-verdt-12-milliarder-kroner/
            "reservoirs": ["svean"],
        },  # RSK development. Assumed flat 2x of the entire system with +5% increased efficiency
        "suldal_i": {
            "eff": 1.05,
            "uprate": 2,
            "reservoirs": ["suldal_1"],
        },
        "suldal_ii": {
            "eff": 1.05,
            "uprate": 2,
            "reservoirs": ["suldal_2"],
        },
        "roeldal": {
            "eff": 1.05,
            "uprate": 2,
            "reservoirs": ["roeldal"],
        },
        "kvanndal": {
            "eff": 1.05,
            "uprate": 2,
            "reservoirs": ["kvanndal"],
        },
        "novle": {
            "eff": 1.05,
            "uprate": 2,
            "reservoirs": ["novle"],
        },
        "svandalsflona": {
            "eff": 1.05,
            "uprate": 2,
            "reservoirs": ["svandalsflon"],
        },
    }

    uprate_hydropower(config=config, uprate_values=uprate_values)

    # %% Used to find the power plants and associated reservoirs

    def write_plants_and_reservoirs(config, plants):
        for plant in plants:
            print(f"==============={plant}===============")
            plant_objs = [i for i in config.plants.find(plant) if plant == i.name]
            for p in plant_objs:
                print(f"***********{p.name}***********")
                format_plant(p)
                connections = find_connections_upper(config, connection_search=p.name)
                print("===========Connections===========")
                for c in connections:
                    print(c)

                print("\n")
                connections = find_connections_upper(config, p.name)
                reservoir_names = [c.from_name for c in connections if c.from_type == ObjectType.RESERVOIR]
                reservoir_objs = []
                for r in reservoir_names:
                    reservoir_objs += [i for i in config.reservoirs.find(r) if r in i.name]

                for r in reservoir_objs:
                    format_reservoir(r)
                print("\n")

    plants_2 = ["hylen", "middyr", "saurdal"]
    plants = list(uprate_values.keys())
    f = io.StringIO()
    with redirect_stdout(f):
        write_plants_and_reservoirs(config=config, plants=plants_2)

    print(f.getvalue())
    # %%
    reservoir = "Numedal"
    reservoir_objs = [i for i in config.reservoirs.find(reservoir) if reservoir in i.name]
    for r in reservoir_objs:
        format_reservoir(r)

    # %%

    pump_name = "kvi"
    pumps = config.pumps.find(pump_name)
    for p in pumps:
        format_pump(p)

    # %%

    plant = list(uprate_values.keys())[0]
    reservoir_name = uprate_values[plant]["reservoirs"][0]
    plant_obj = [i for i in config.plants.find(plant) if i.name == plant][0]
    reservoir_obj = [i for i in config.reservoirs.find(reservoir_name) if i.name == reservoir_name][0]
    format_plant(plant_obj)
    format_reservoir(reservoir_obj)
    reservoir_obj.metadata.origin_area

    plant_connections = []
    reservoir_connections = []
    for con in config.connections:
        if con.from_name == plant_obj.name or con.to_name == plant_obj.name:
            plant_connections.append(con)
        if con.from_name == reservoir_name or con.to_name == reservoir_name:
            reservoir_connections.append(con)

    for hc in config.hydraulic_couplings:
        hc

    plant_obj.metadata.watercourse

    # Initialize nested defaultdict
    hydro_data = defaultdict(
        lambda: {"Reservoir volume [Mm3]": 0.0, "Hydro capacity [MW]": 0.0, "Reservoir volume [TWh]": 0.0}
    )

    # Loop over all reservoirs
    for res in config.reservoirs:
        area = res.metadata.origin_area
        hydro_data[area]["Reservoir volume [Mm3]"] += res.metadata.reservoir_capacity_mm3
        hydro_data[area]["Reservoir volume [TWh]"] += (
            res.metadata.reservoir_capacity_mm3 * res.metadata.global_energy_equivalent / 1e3
        )

        # Find connections from reservoir to plants
        for con in config.connections:
            if res.name == con.from_name and con.to_type == ObjectType.PLANT:
                plant_objs = config.plants.find(con.to_name)
                if plant_objs:
                    plant_obj = plant_objs[0]
                    # Assume last point of PQ curve gives maximum capacity
                    if plant_obj.pq_curves and plant_obj.pq_curves.config:
                        capacity = plant_obj.pq_curves.config.y[-1]
                        hydro_data[area]["Hydro capacity [MW]"] += capacity

    import pandas as pd

    added_capacity_per_area = get_uprated_capacity_by_area(config, uprate_values)

    # Convert hydro data to DataFrame
    df = pd.DataFrame.from_dict(hydro_data, orient="index")

    # Add volume and capacity shares (before uprate)
    df["Volume share [%]"] = 100 * df["Reservoir volume [TWh]"] / df["Reservoir volume [TWh]"].sum()
    df["Capacity share [%]"] = 100 * df["Hydro capacity [MW]"] / df["Hydro capacity [MW]"].sum()

    # Add uprate index
    uprate_index = defaultdict(float)
    for value in uprate_values.values():
        uprate_index[value["elspot_area"]] += value["uprate"]

    # Add added capacity and new capacity
    df["Added capacity [MW]"] = df.index.map(lambda x: added_capacity_per_area.get(x, 0.0))
    df["New hydro capacity [MW]"] = df["Hydro capacity [MW]"] + df["Added capacity [MW]"]

    # Add new capacity share
    df["New capacity share [%]"] = 100 * df["New hydro capacity [MW]"] / df["Hydro capacity [MW]"].sum()

    # Optional: Add total row
    total = pd.DataFrame(df.sum(numeric_only=True)).T
    total.index = ["Total"]
    total["Volume share [%]"] = 100.0  # Total system share
    total["Capacity share [%]"] = 100.0
    total["New capacity share [%]"] = 100.0
    df = pd.concat([df, total])

    df = df.drop(columns=["New hydro capacity [MW]", "Reservoir volume [Mm3]"])
    # Sort for consistent output
    df = df.sort_index()

    def format_df_for_latex(df):
        df_formatted = df.copy()

        # Format integer values
        for col in ["Hydro capacity [MW]", "Added capacity [MW]"]:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")

        for col in ["Reservoir volume [TWh]"]:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

        # Format percentages
        for col in ["Volume share [%]", "Capacity share [%]", "New capacity share [%]"]:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}\\%" if pd.notnull(x) else "")

        return df_formatted

    df = format_df_for_latex(df)

    df = df[
        [
            "Reservoir volume [TWh]",
            "Volume share [%]",
            "Hydro capacity [MW]",
            "Capacity share [%]",
            "Added capacity [MW]",
            "New capacity share [%]",
        ]
    ]
    # Generate LaTeX table
    df.columns = [col.replace("%", r"\%") for col in df.columns]
    latex_table = df.to_latex(
        float_format="%.2f",
        index=True,
        caption="Hydropower capacity and reservoir volumes per elspot area, before and after uprates.",
        label="tab:hydro_area_uprate",
        column_format="lrrrrrrr",
        escape=False,
    )

    # Output LaTeX
    print(latex_table)


# no = [f"NO{i}" for i in range(1, 6)]

# df[df.index.isin(no)].sum()

# %%


def test():
    df_renewables_profiles = pd.read_parquet("data/renewables_profiles.parquet")
    config = None
    add_area_with_offshore_wind(
        config=config,
        new_area_name="Sydvest_A",
        connected_area="NO2",
        transmission_capacity_mw=1500,
        offshore_profile_name="NO2_wind_offshore_Sydvest_A",  # Must match column in df_renewables_profiles
        offshore_pmax=1500,
        df_renewables_profiles=df_renewables_profiles,
        logger=_logger,
    )
