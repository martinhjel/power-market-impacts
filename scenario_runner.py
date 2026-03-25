from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List

from dataset_runner import LoadMode, run_dataset

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    name: str
    resolution: str = "1D"
    load_mode: LoadMode = LoadMode.NONE
    additional_load_twh: float = 0.0
    uprate_hydro: bool = False
    update_feedback_factors: bool = False
    nuclear_additions: List[dict] = field(default_factory=list)
    offshore_wind_additions: List[dict] = field(default_factory=list)


SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        name="BASELINE_00TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=0.0,
        uprate_hydro=False,
        nuclear_additions=[],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="BASELINE_10TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=10.0,
        uprate_hydro=False,
        nuclear_additions=[],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="BASELINE_20TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=20.0,
        uprate_hydro=False,
        nuclear_additions=[],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="BASELINE_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="BASELINE_40TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=40.0,
        uprate_hydro=False,
        nuclear_additions=[],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="BASELINE_50TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=50.0,
        uprate_hydro=False,
        nuclear_additions=[],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="BASELINE_UPRATE",
        resolution="1H",
        load_mode=LoadMode.NONE,
        additional_load_twh=0.0,
        uprate_hydro=True,
        nuclear_additions=[],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="LLPS_OW",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=22.910413440000003,
        uprate_hydro=False,
        offshore_wind_additions=[
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_SorligeNordsjo2",
                "capacity": 3000,
                "new_area_name": "SNII",
            },
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_UtsiraNord",
                "capacity": 500,
                "new_area_name": "UN",
            },
            {
                "connected_to": "NO5",
                "profile": "NO5_wind_offshore_Vestavind_D",
                "capacity": 1500,
                "new_area_name": "VVD",
            },
        ],
    ),
    ScenarioConfig(
        name="LLPS_N",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=22.910413440000003,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO2", "capacity": 900.8407111111112, "price": 9},
            {"area": "NO1", "capacity": 2005.0970666666665, "price": 9},
        ],
    ),
    ScenarioConfig(
        name="LLPS_OWN",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=22.910413440000003,
        uprate_hydro=False,
        nuclear_additions=[{"area": "NO1", "capacity": 1781.3768888888892, "price": 9}],
        offshore_wind_additions=[
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_SorligeNordsjo2",
                "capacity": 1400,
                "new_area_name": "SNII",
            },
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_UtsiraNord",
                "capacity": 500,
                "new_area_name": "UN",
            },
        ],
    ),
    ScenarioConfig(
        name="BA_OW",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=22.910413440000003,
        uprate_hydro=False,
        offshore_wind_additions=[
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_SorligeNordsjo2",
                "capacity": 3000,
                "new_area_name": "SNII",
            },
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_UtsiraNord",
                "capacity": 500,
                "new_area_name": "UN",
            },
            {
                "connected_to": "NO5",
                "profile": "NO5_wind_offshore_Vestavind_D",
                "capacity": 1500,
                "new_area_name": "VVD",
            },
        ],
    ),
    ScenarioConfig(
        name="BA_N",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=22.910413440000003,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO2", "capacity": 900.8407111111112, "price": 9},
            {"area": "NO1", "capacity": 2005.0970666666665, "price": 9},
        ],
    ),
    ScenarioConfig(
        name="BA_OWN",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=22.910413440000003,
        uprate_hydro=False,
        nuclear_additions=[{"area": "NO1", "capacity": 1781.3768888888892, "price": 9}],
        offshore_wind_additions=[
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_SorligeNordsjo2",
                "capacity": 1400,
                "new_area_name": "SNII",
            },
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_UtsiraNord",
                "capacity": 500,
                "new_area_name": "UN",
            },
        ],
    ),
    ScenarioConfig(
        name="LLPS_OW_UPRATE",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=22.910413440000003,
        uprate_hydro=True,
        offshore_wind_additions=[
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_SorligeNordsjo2",
                "capacity": 3000,
                "new_area_name": "SNII",
            },
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_UtsiraNord",
                "capacity": 500,
                "new_area_name": "UN",
            },
            {
                "connected_to": "NO5",
                "profile": "NO5_wind_offshore_Vestavind_D",
                "capacity": 1500,
                "new_area_name": "VVD",
            },
        ],
    ),
    ScenarioConfig(
        name="LLPS_N_UPRATE",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=22.910413440000003,
        uprate_hydro=True,
        nuclear_additions=[
            {"area": "NO2", "capacity": 900.8407111111112, "price": 9},
            {"area": "NO1", "capacity": 2005.0970666666665, "price": 9},
        ],
    ),
    ScenarioConfig(
        name="LLPS_OWN_UPRATE",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=22.910413440000003,
        uprate_hydro=True,
        nuclear_additions=[{"area": "NO1", "capacity": 1781.3768888888892, "price": 9}],
        offshore_wind_additions=[
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_SorligeNordsjo2",
                "capacity": 1400,
                "new_area_name": "SNII",
            },
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_UtsiraNord",
                "capacity": 500,
                "new_area_name": "UN",
            },
        ],
    ),
    ScenarioConfig(
        name="BA_OW_UPRATE",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=22.910413440000003,
        uprate_hydro=True,
        offshore_wind_additions=[
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_SorligeNordsjo2",
                "capacity": 3000,
                "new_area_name": "SNII",
            },
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_UtsiraNord",
                "capacity": 500,
                "new_area_name": "UN",
            },
            {
                "connected_to": "NO5",
                "profile": "NO5_wind_offshore_Vestavind_D",
                "capacity": 1500,
                "new_area_name": "VVD",
            },
        ],
    ),
    ScenarioConfig(
        name="BA_N_UPRATE",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=22.910413440000003,
        uprate_hydro=True,
        nuclear_additions=[
            {"area": "NO2", "capacity": 900.8407111111112, "price": 9},
            {"area": "NO1", "capacity": 2005.0970666666665, "price": 9},
        ],
    ),
    ScenarioConfig(
        name="BA_OWN_UPRATE",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=22.910413440000003,
        uprate_hydro=True,
        nuclear_additions=[{"area": "NO1", "capacity": 1781.3768888888892, "price": 9}],
        offshore_wind_additions=[
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_SorligeNordsjo2",
                "capacity": 1400,
                "new_area_name": "SNII",
            },
            {
                "connected_to": "NO2",
                "profile": "NO2_wind_offshore_UtsiraNord",
                "capacity": 500,
                "new_area_name": "UN",
            },
        ],
    ),
    ScenarioConfig(
        name="SMR300BA_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 300, "price": 9},
            {"area": "NO2", "capacity": 300, "price": 9},
            {"area": "NO3", "capacity": 300, "price": 9},
            {"area": "NO4", "capacity": 300, "price": 9},
            {"area": "NO5", "capacity": 300, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR600BA_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 600, "price": 9},
            {"area": "NO2", "capacity": 600, "price": 9},
            {"area": "NO3", "capacity": 600, "price": 9},
            {"area": "NO4", "capacity": 600, "price": 9},
            {"area": "NO5", "capacity": 600, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR900BA_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 900, "price": 9},
            {"area": "NO2", "capacity": 900, "price": 9},
            {"area": "NO3", "capacity": 900, "price": 9},
            {"area": "NO4", "capacity": 900, "price": 9},
            {"area": "NO5", "capacity": 900, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR1200BA_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 1200, "price": 9},
            {"area": "NO2", "capacity": 1200, "price": 9},
            {"area": "NO3", "capacity": 1200, "price": 9},
            {"area": "NO4", "capacity": 1200, "price": 9},
            {"area": "NO5", "capacity": 1200, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR1600BA_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 1600, "price": 9},
            {"area": "NO2", "capacity": 1600, "price": 9},
            {"area": "NO3", "capacity": 1600, "price": 9},
            {"area": "NO4", "capacity": 1600, "price": 9},
            {"area": "NO5", "capacity": 1600, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="LMR2000BA_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 2000, "price": 9},
            {"area": "NO2", "capacity": 2000, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="LMR3000BA_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 3000, "price": 9},
            {"area": "NO2", "capacity": 3000, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="LMR4000BA_30TWh",
        resolution="1H",
        load_mode=LoadMode.BA,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 4000, "price": 9},
            {"area": "NO2", "capacity": 4000, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR300LLPS_30TWh",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 300, "price": 9},
            {"area": "NO2", "capacity": 300, "price": 9},
            {"area": "NO3", "capacity": 300, "price": 9},
            {"area": "NO4", "capacity": 300, "price": 9},
            {"area": "NO5", "capacity": 300, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR600LLPS_30TWh",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 600, "price": 9},
            {"area": "NO2", "capacity": 600, "price": 9},
            {"area": "NO3", "capacity": 600, "price": 9},
            {"area": "NO4", "capacity": 600, "price": 9},
            {"area": "NO5", "capacity": 600, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR900LLPS_30TWh",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 900, "price": 9},
            {"area": "NO2", "capacity": 900, "price": 9},
            {"area": "NO3", "capacity": 900, "price": 9},
            {"area": "NO4", "capacity": 900, "price": 9},
            {"area": "NO5", "capacity": 900, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR1200LLPS_30TWh",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 1200, "price": 9},
            {"area": "NO2", "capacity": 1200, "price": 9},
            {"area": "NO3", "capacity": 1200, "price": 9},
            {"area": "NO4", "capacity": 1200, "price": 9},
            {"area": "NO5", "capacity": 1200, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="SMR1600LLPS_30TWh",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 1600, "price": 9},
            {"area": "NO2", "capacity": 1600, "price": 9},
            {"area": "NO3", "capacity": 1600, "price": 9},
            {"area": "NO4", "capacity": 1600, "price": 9},
            {"area": "NO5", "capacity": 1600, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="LMR2000LLPS_30TWh",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 2000, "price": 9},
            {"area": "NO2", "capacity": 2000, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="LMR3000LLPS_30TWh",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 3000, "price": 9},
            {"area": "NO2", "capacity": 3000, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
    ScenarioConfig(
        name="LMR4000LLPS_30TWh",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        additional_load_twh=30.0,
        uprate_hydro=False,
        nuclear_additions=[
            {"area": "NO1", "capacity": 4000, "price": 9},
            {"area": "NO2", "capacity": 4000, "price": 9},
        ],
        offshore_wind_additions=[],
    ),
]


def run_single_scenario(scenario):
    logger.info("Running scenario %s", scenario.name)
    try:
        destination = run_dataset(
            scenario_name=scenario.name,
            resolution=scenario.resolution,
            load_mode=scenario.load_mode,
            new_yearly_load_twh=scenario.additional_load_twh,
            do_uprate_hydropower=scenario.uprate_hydro,
            do_update_feedback_factors=scenario.update_feedback_factors,
            nuclear_additions=scenario.nuclear_additions,
            offshore_wind_additions=scenario.offshore_wind_additions,
        )
        logger.info("Scenario %s completed. Results in %s", scenario.name, destination)
        return (scenario.name, destination)
    except Exception as exc:
        logger.exception("Scenario %s failed: %s", scenario.name, exc)
        return (scenario.name, exc)


def run_scenarios_parallel(max_workers=None):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_scenario = {executor.submit(run_single_scenario, scenario): scenario for scenario in SCENARIOS}
        for future in as_completed(future_to_scenario):
            result = future.result()
            results.append(result)
    return results


if __name__ == "__main__":
    n_procs = 8
    results = run_scenarios_parallel(max_workers=n_procs)
    print("Finished:", results)
