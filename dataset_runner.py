import logging
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from cognite.client.testing import CogniteClientMock
from lpr_sintef_bifrost.models import EMPSModelBuilder

from dataset_adjuster import (
    add_area_with_offshore_wind,
    add_load_as_ba,
    add_load_as_llps,
    add_nuclear,
    update_feedback_factors,
    uprate_hydropower,
)
from nuclear_modeling import (
    HISTORIC_NUCLEAR_PROFILE_PATH,
    IMPROVE_NUCLEAR_REP,
    NEW_NUCLEAR_FIRM_SHARE,
    NEW_NUCLEAR_MARGINAL_COST,
    NEW_NUCLEAR_PROFILE_PATH,
    improved_nuclear_output_suffix,
)

__all__ = ["LoadMode", "run_dataset"]


class LoadMode(Enum):
    LLPS = "LLPS"
    BA = "BA"
    NONE = "NONE"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger("lpr_sintef_bifrost")


def _format_capacity(value) -> str:
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", "p")
    return str(value)


def _format_addition_label(additions: Iterable[dict], area_key: str) -> str:
    items = list(additions or [])
    if not items:
        return "None"

    labels = []
    for addition in items:
        area = addition.get(area_key, "NA")
        capacity = _format_capacity(addition.get("capacity", "NA"))
        labels.append(f"{capacity}{area}")
    return "-".join(labels)


def _sanitize_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label)


def _build_destination_name(
    *,
    scenario_name: Optional[str],
    do_uprate_hydropower: bool,
    do_update_feedback_factors: bool,
    load_mode: LoadMode,
    new_yearly_load_twh: float,
    nuclear_additions: Iterable[dict],
    offshore_wind_additions: Iterable[dict],
) -> str:
    labels = []
    if scenario_name:
        labels.append(_sanitize_label(scenario_name))

    labels.append(f"{do_uprate_hydropower}HYD")
    labels.append(f"{do_update_feedback_factors}FF")
    labels.append(f"{load_mode.value}LOAD")
    labels.append(f"{new_yearly_load_twh:.2f}TWH")
    labels.append(f"{_format_addition_label(nuclear_additions, 'area')}NUKE")
    labels.append(f"{_format_addition_label(offshore_wind_additions, 'connected_to')}OFF")

    return "_".join(labels)


def run_dataset(
    *,
    scenario_name: Optional[str] = None,
    resolution: str = "1D",
    do_uprate_hydropower: bool = False,
    new_yearly_load_twh: float = 0.0,
    load_mode: LoadMode = LoadMode.NONE,
    do_update_feedback_factors: bool = False,
    nuclear_additions: Optional[list[dict]] = None,
    offshore_wind_additions: Optional[list[dict]] = None,
    dataset_year: int = 2025,
    dataset_scenario: str = "BM",
    dataset_version: str = "100",
    model_name: str = "PowerGamaMSc",
    simulation_type: str = "serial",
    use_exogenous_prices: bool = True,
    destination_root: Path | None = None,
    clear_destination_folder: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    convert_large_timeseries_to_hdf5: bool = True,
    export_detailed_results: bool = False,
    n_cpu: int = 1,
    cdf_client=None,
    dataset_folder: Path | None = None,
    dataset_file: Path | None = None,
    renewables_profile_path: Path | str = Path("data/renewables_profiles.parquet"),
    improve_nuclear_rep: bool = IMPROVE_NUCLEAR_REP,
    new_nuclear_firm_share: float = NEW_NUCLEAR_FIRM_SHARE,
) -> Path:
    """Run a single dataset adjustment and simulation."""

    nuclear_additions = nuclear_additions or []
    offshore_wind_additions = offshore_wind_additions or []
    cdf_client = cdf_client or CogniteClientMock()

    if dataset_folder is None:
        dataset_folder = (
            Path.cwd()
            / (
                f"ltm_output/{model_name}_{dataset_year}_{dataset_scenario}_{resolution}_"
                f"{simulation_type}_{use_exogenous_prices}EXO_load"
                f"{improved_nuclear_output_suffix(improve_nuclear_rep)}/"
            )
        )
    dataset_folder.mkdir(parents=True, exist_ok=True)

    if dataset_file is None:
        dataset_file = dataset_folder / "dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found.")

    if destination_root is None:
        destination_root = dataset_folder

    destination_name = _build_destination_name(
        scenario_name=scenario_name,
        do_uprate_hydropower=do_uprate_hydropower,
        do_update_feedback_factors=do_update_feedback_factors,
        load_mode=load_mode,
        new_yearly_load_twh=new_yearly_load_twh,
        nuclear_additions=nuclear_additions,
        offshore_wind_additions=offshore_wind_additions,
    )
    destination_folder = destination_root / destination_name
    destination_folder.mkdir(parents=True, exist_ok=True)

    for handler in list(_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            _logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(str(destination_folder / "log.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(file_handler)

    _logger.info(f"Reading model from {dataset_file}")
    config = EMPSModelBuilder.from_json(filepath=dataset_file)
    _logger.info("Finished reading model!")

    if do_uprate_hydropower:
        import sys

        sys.path.append(str(Path.cwd() / "data"))
        from uprate_hydro import uprate_values  # type: ignore

        uprate_hydropower(config, uprate_values=uprate_values)

    for nuclear_entry in nuclear_additions:
        node = nuclear_entry["area"]
        capacity = nuclear_entry["capacity"]
        price = nuclear_entry.get("price", 9)
        add_nuclear(
            config=config,
            node=node,
            capacity=capacity,
            price=price,
            improve_nuclear_rep=improve_nuclear_rep,
            firm_share=new_nuclear_firm_share,
        )

    if do_update_feedback_factors:
        update_feedback_factors(config)

    if load_mode == LoadMode.LLPS:
        _logger.info(f"Adding {new_yearly_load_twh:.2f} TWh load using LLPS.")
        add_load_as_llps(config, new_yearly_load_twh)
    elif load_mode == LoadMode.BA:
        _logger.info(f"Adding {new_yearly_load_twh:.2f} TWh load as baseload.")
        add_load_as_ba(config, new_yearly_load_twh)

    if offshore_wind_additions:
        df_renewables_profiles = pd.read_parquet(renewables_profile_path)
        for idx, offshore in enumerate(offshore_wind_additions, start=1):
            new_area_name = (
                offshore.get("new_area_name")
                or offshore.get("new_area")
                or f"{offshore['connected_to']}_offshore_{idx}"
            )
            add_area_with_offshore_wind(
                config=config,
                new_area_name=new_area_name,
                connected_area=offshore["connected_to"],
                transmission_capacity_mw=offshore["capacity"],
                offshore_profile_name=offshore["profile"],
                offshore_pmax=offshore.get("pmax", offshore["capacity"]),
                df_renewables_profiles=df_renewables_profiles,
                logger=_logger,
            )

    _logger.info(f"Setting num_processes to {n_cpu}.")
    config.global_settings.num_processes_override = n_cpu
    config.cdf_client = cdf_client

    # Close the file handler before config.run() to prevent log.log from being deleted
    # when clear_destination_folder=True
    _logger.info("Closing log handler before config.run()")
    if file_handler in _logger.handlers:
        _logger.removeHandler(file_handler)
        file_handler.close()

    try:
        config.run(
            destination_folder=destination_folder,
            resolution=resolution,
            clear_destination_folder=clear_destination_folder,
            dry_run=dry_run,
            verbose=verbose,
            convert_large_timeseries_to_hdf5=convert_large_timeseries_to_hdf5,
            export_detailed_results=export_detailed_results,
        )
    except Exception:
        failure_handler = logging.FileHandler(str(destination_folder / "log.log"), mode="a")
        failure_handler.setLevel(logging.INFO)
        failure_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        _logger.addHandler(failure_handler)
        _logger.exception("Scenario failed during config.run")
        _logger.removeHandler(failure_handler)
        failure_handler.close()
        raise

    # Reopen the log file handler in append mode to continue logging
    file_handler = logging.FileHandler(str(destination_folder / "log.log"), mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(file_handler)

    _logger.info(f"Scenario complete. Results written to {destination_folder}")

    run_config = {
        "scenario_name": scenario_name,
        "improve_nuclear_rep": improve_nuclear_rep,
        "historic_nuclear_profile": HISTORIC_NUCLEAR_PROFILE_PATH,
        "new_nuclear_profile": NEW_NUCLEAR_PROFILE_PATH,
        "new_nuclear_firm_share": new_nuclear_firm_share if improve_nuclear_rep else 0.0,
        "new_nuclear_flexible_share": 1.0 - (new_nuclear_firm_share if improve_nuclear_rep else 0.0),
        "new_nuclear_marginal_cost": sorted({entry.get("price", NEW_NUCLEAR_MARGINAL_COST) for entry in nuclear_additions}),
        "nuclear_market_step_capacity_scenario_independent": True,
    }
    with open(destination_folder / "config.txt", "w") as f:
        for key, value in run_config.items():
            f.write(f"{key}: {value}\n")

    # Clean up handler at the end
    _logger.removeHandler(file_handler)
    file_handler.close()

    return destination_folder


if __name__ == "__main__":
    # run_dataset(scenario_name="default")

    run_dataset(
        scenario_name="fixed_nuclear",
        resolution="1H",
        load_mode=LoadMode.LLPS,
        new_yearly_load_twh=20.0,
        do_uprate_hydropower=False,
        nuclear_additions=[{"area": "NO2", "capacity": 499, "price": 9}],
        n_cpu=1,
    )
