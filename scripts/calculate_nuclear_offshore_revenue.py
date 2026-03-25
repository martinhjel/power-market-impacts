"""
Calculate revenue for nuclear and offshore wind resources in BA_N, BA_OW, and BA_OWN scenarios.

This script:
1. Loads scenario results for N (Nuclear), OW (Offshore Wind), and OWN (Offshore Wind + Nuclear) cases
2. Extracts nuclear generation using market steps (similar to viz_nuclear_factors.py)
3. Extracts offshore wind generation from wind objects in areas SNII, UN, and VVD
4. Calculates revenue = generation * price for each technology
5. Computes value factors and capacity factors
6. Computes break-even CAPEX at construction start (FID)
7. Outputs results to CSV and creates visualizations
"""

import logging
import re
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Add workspace root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import logger

# Set logger level to DEBUG to see debug messages
logger.setLevel(logging.DEBUG)

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
NUCLEAR_PRICE = 9.0  # EUR/MWh - nuclear bid price
NUCLEAR_OPEX = 26.4  # EUR/MWh - nuclear operating cost
OFFSHORE_WIND_OPEX = 24.2  # EUR/MWh - offshore wind operating cost
NUCLEAR_PROFILE_PATH = Path.cwd() / "data" / "new_nuclear_profile.parquet"
_nuclear_profile_df = pd.read_parquet(NUCLEAR_PROFILE_PATH)
NUCLEAR_PROFILE_CF = (
    _nuclear_profile_df["capacity_factor"].mean()
    if "capacity_factor" in _nuclear_profile_df.columns
    else _nuclear_profile_df.iloc[:, 0].mean()
)
OFFSHORE_PROFILE_PATH = Path.cwd() / "data" / "renewables_profiles.parquet"
OFFSHORE_PROFILE_COLUMN_BY_AREA = {
    "SNII": "NO2_wind_offshore_SorligeNordsjo2",
    "UN": "NO2_wind_offshore_UtsiraNord",
    "VVD": "NO5_wind_offshore_Vestavind_D",
}
_offshore_profile_df = pd.read_parquet(
    OFFSHORE_PROFILE_PATH, columns=list(OFFSHORE_PROFILE_COLUMN_BY_AREA.values())
)
OFFSHORE_PROFILE_CF_BY_AREA = {
    area: _offshore_profile_df[column].mean() for area, column in OFFSHORE_PROFILE_COLUMN_BY_AREA.items()
}

# Annuity factors for CAPEX calculation (assuming 5% discount rate)
NUCLEAR_LIFETIME = 60  # years
OFFSHORE_WIND_LIFETIME = 25  # years
DISCOUNT_RATE = 0.05
NUCLEAR_CONSTRUCTION_YEARS = 5
OFFSHORE_WIND_CONSTRUCTION_YEARS = 4
# Annuity factor = [1 - (1 + r)^-n] / r
NUCLEAR_ANNUITY_FACTOR = (1 - (1 + DISCOUNT_RATE) ** -NUCLEAR_LIFETIME) / DISCOUNT_RATE
OFFSHORE_WIND_ANNUITY_FACTOR = (1 - (1 + DISCOUNT_RATE) ** -OFFSHORE_WIND_LIFETIME) / DISCOUNT_RATE

AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
OFFSHORE_WIND_AREAS = ["SNII", "UN", "VVD"]  # Offshore wind is in separate busbars
CONNECTED_OFFSHORE_AREAS = {
    "SNII": "NO2",
    "UN": "NO2",
    "VVD": "NO5",
}

SCENARIOS = {
    "N-LLPS+": "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-LLPS+": "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-LLPS+": "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "N-BA+": "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-BA+": "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-BA+": "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "N-LLPS": "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-LLPS": "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-LLPS": "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "N-BA": "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-BA": "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-BA": "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    # "B": "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF",
    "SMR300-BA": "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR300-LLPS": "SMR300LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR600-BA": "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR600-LLPS": "SMR600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR900-BA": "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR900-LLPS": "SMR900LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR1200-BA": "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1200-LLPS": "SMR1200LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1600-BA": "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    "SMR1600-LLPS": "SMR1600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    "LMR2000-BA": "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR2000-LLPS": "LMR2000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR3000-BA": "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR3000-LLPS": "LMR3000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR4000-BA": "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
    "LMR4000-LLPS": "LMR4000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
}


# Setup paths
base_path = Path.cwd()
# Handle running from scripts/paper or workspace root
if base_path.name == "paper":
    base_path = base_path.parent.parent
elif base_path.name == "scripts":
    base_path = base_path.parent

ltm_output_path = base_path / "ltm_output" / MODEL_FOLDER
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER / "paper"
output_path.mkdir(parents=True, exist_ok=True)


def _load_market_dispatch_data(scenario_path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    processed_dir = scenario_path / "results" / "processed"
    export_file = processed_dir / "market_dispatch.pkl"
    process_file = processed_dir / "market_dispatch.parquet"
    if not export_file.exists() or not process_file.exists():
        logger.warning(f"Missing processed files in {processed_dir}")
        return None
    return pd.read_pickle(export_file), pd.read_parquet(process_file)


def _parse_nuclear_capacity_by_area_from_scenario_name(scenario_name: str) -> dict[str, float]:
    """
    Parse nuclear capacity additions (MW) from scenario folder name.
    Expected segment:
      ..._22.91TWH_900p84NO2-2005p09NO1NUKE_...
    """
    if "TWH_" not in scenario_name or "NUKE_" not in scenario_name:
        return {}

    capacity_segment = scenario_name.split("TWH_", 1)[1].split("NUKE_", 1)[0]
    if capacity_segment == "None":
        return {}

    capacities: dict[str, float] = {}
    for token in capacity_segment.split("-"):
        match = re.match(r"^(?P<cap>[0-9]+(?:p[0-9]+)?)(?P<area>NO[1-5])$", token)
        if match is None:
            continue
        capacities[match.group("area")] = float(match.group("cap").replace("p", "."))
    return capacities


def _parse_offshore_capacity_by_area_from_scenario_name(scenario_name: str) -> dict[str, float]:
    """
    Parse offshore wind capacity additions (MW) from scenario folder name.
    Expected segment:
      ...NUKE_3000NO2-500NO2-1500NO5OFF
    Repeated connected onshore areas are assigned in OFFSHORE_WIND_AREAS order.
    """
    match = re.search(r"NUKE_(.+?)OFF", scenario_name)
    if match is None:
        return {}

    capacity_segment = match.group(1)
    if capacity_segment == "None":
        return {}

    area_slots_by_connected = {
        connected_area: [area for area in OFFSHORE_WIND_AREAS if CONNECTED_OFFSHORE_AREAS[area] == connected_area]
        for connected_area in {CONNECTED_OFFSHORE_AREAS[a] for a in OFFSHORE_WIND_AREAS}
    }
    capacities: dict[str, float] = {}

    for token in capacity_segment.split("-"):
        token_match = re.match(r"^(?P<cap>[0-9]+(?:p[0-9]+)?)(?P<connected>NO[1-5])$", token)
        if token_match is None:
            continue
        connected_area = token_match.group("connected")
        slots = area_slots_by_connected.get(connected_area)
        if not slots:
            continue
        offshore_area = slots.pop(0)
        capacities[offshore_area] = float(token_match.group("cap").replace("p", "."))

    return capacities


def _expected_nuclear_generation_gwh(capacity_mw: float, n_hours_per_year: float) -> float:
    if capacity_mw <= 0 or n_hours_per_year <= 0:
        return 0.0
    return capacity_mw * NUCLEAR_PROFILE_CF * n_hours_per_year / 1000.0


def _parse_target_twh_from_scenario_name(scenario_name: str) -> float | None:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)TWH_", scenario_name)
    if match is None:
        return None
    return float(match.group(1))


def get_nuclear_generation_and_revenue(
    export_df: pd.DataFrame,
    process_df: pd.DataFrame,
    area: str,
    nuclear_price: float,
    scenario_name: str | None = None,
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Extract nuclear generation and calculate revenue for a specific area.

    Returns:
        df_nuke_generation: DataFrame with nuclear generation (MW)
        total_revenue: Total revenue (EUR)
        capacity_factor: Nuclear capacity factor
        value_factor: Nuclear value factor
    """
    if area not in process_df.index.get_level_values("area"):
        return None, 0.0, 0.0, 0.0

    df_area = process_df.xs(area, level="area")
    if "nuclear" not in df_area.columns:
        return None, 0.0, 0.0, 0.0

    df_nuke_generation = df_area[["nuclear"]]
    df_price = df_area[["market_price"]]

    n_weather_years = len(df_nuke_generation.index.get_level_values("scenario").unique())
    total_revenue = (df_nuke_generation["nuclear"] * df_price["market_price"]).sum() / n_weather_years

    scenarios = df_nuke_generation.index.get_level_values("scenario")
    if scenario_name is not None:
        installed_capacity_mw = (
            _parse_nuclear_capacity_by_area_from_scenario_name(scenario_name).get(area, 0.0) * len(scenarios)
        )
    else:
        scenario_capacity_by_name = {
            scenario: _parse_nuclear_capacity_by_area_from_scenario_name(str(scenario)).get(area, 0.0)
            for scenario in scenarios.unique()
        }
        installed_capacity_mw = scenarios.map(scenario_capacity_by_name).sum()
    capacity_factor = (
        df_nuke_generation["nuclear"].sum() / installed_capacity_mw if installed_capacity_mw > 0 else 0.0
    )

    achieved_price = (
        (df_price["market_price"] * df_nuke_generation["nuclear"]).sum() / df_nuke_generation["nuclear"].sum()
        if df_nuke_generation["nuclear"].sum() > 0
        else 0.0
    )
    weighted_price = df_price["market_price"].mean()
    value_factor = achieved_price / weighted_price if weighted_price > 0 else 0.0

    return df_nuke_generation, total_revenue, capacity_factor, value_factor


def get_offshore_wind_generation_and_revenue(
    export_df: pd.DataFrame, process_df: pd.DataFrame, area: str, scenario_name: str | None = None
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Extract offshore wind generation and calculate revenue for offshore wind areas (SNII, UN, VVD).

    Returns:
        df_wind_generation: DataFrame with offshore wind generation (MW)
        total_revenue: Total revenue (EUR)
        capacity_factor: Offshore wind capacity factor
        value_factor: Offshore wind value factor
        curtailed_generation: Total curtailed generation (GWh)
    """
    if area not in export_df.index.get_level_values("area"):
        return None, 0.0, 0.0, 0.0, 0.0
    if area not in process_df.index.get_level_values("area"):
        return None, 0.0, 0.0, 0.0, 0.0

    df_wind_generation = export_df.xs(area, level="area")[["offshore_wind"]]
    df_price = process_df.xs(area, level="area")[["market_price"]]

    n_weather_years = len(df_wind_generation.index.get_level_values("scenario").unique())
    total_revenue = (df_wind_generation["offshore_wind"] * df_price["market_price"]).sum() / n_weather_years

    scenarios = df_wind_generation.index.get_level_values("scenario")
    if scenario_name is not None:
        installed_capacity_mw = (
            _parse_offshore_capacity_by_area_from_scenario_name(scenario_name).get(area, 0.0) * len(scenarios)
        )
    else:
        scenario_capacity_by_name = {
            scenario: _parse_offshore_capacity_by_area_from_scenario_name(str(scenario)).get(area, 0.0)
            for scenario in scenarios.unique()
        }
        installed_capacity_mw = scenarios.map(scenario_capacity_by_name).sum()
    curtailed_gwh = (
        df_wind_generation.join(export_df.xs(area, level="area")[["market_steps"]])["market_steps"]
        .clip(upper=0.0)
        .abs()
        .sum()
        / n_weather_years
        / 1000.0
    )
    capacity_factor = (
        df_wind_generation["offshore_wind"].sum() / installed_capacity_mw if installed_capacity_mw > 0 else 0.0
    )

    achieved_price = (
        (df_price["market_price"] * df_wind_generation["offshore_wind"]).sum()
        / df_wind_generation["offshore_wind"].sum()
        if df_wind_generation["offshore_wind"].sum() > 0
        else 0.0
    )
    weighted_price = df_price["market_price"].mean()
    value_factor = achieved_price / weighted_price if weighted_price > 0 else 0.0

    return df_wind_generation, total_revenue, capacity_factor, value_factor, curtailed_gwh


logger.info("Starting nuclear and offshore wind revenue calculation")
logger.info(f"Analyzing scenarios: {list(SCENARIOS.keys())}")

# Initialize results storage
results = []

def _breakeven_capex_fid_per_kw(
    annual_net_revenue_meur: float,
    capacity_kw: float,
    annuity_factor: float,
    construction_years: int,
    discount_rate: float,
) -> float:
    """
    Break-even CAPEX at construction start (FID), EUR/kW.
    Revenue starts after construction completion.
    """
    if capacity_kw <= 0:
        return 0.0

    pv_revenue_meur_at_fid = (annual_net_revenue_meur * annuity_factor) / ((1 + discount_rate) ** construction_years)
    return (pv_revenue_meur_at_fid * 1e6) / capacity_kw


# Process each scenario
for scenario_label, scenario_name in SCENARIOS.items():
    scenario_path = ltm_output_path / scenario_name

    if not scenario_path.exists():
        logger.warning(f"Scenario path does not exist: {scenario_path}")
        continue

    data = _load_market_dispatch_data(scenario_path)
    if data is None:
        logger.warning(f"Missing market dispatch files for {scenario_label}")
        continue
    export_df, process_df = data
    nuclear_capacity_by_area = _parse_nuclear_capacity_by_area_from_scenario_name(scenario_name)
    target_twh = _parse_target_twh_from_scenario_name(scenario_name)

    logger.info(f"\nProcessing {scenario_label}: {scenario_name}")
    scenario_nuclear_generation_gwh = 0.0
    scenario_nuclear_curtailed_gwh = 0.0
    scenario_offshore_generation_gwh = 0.0
    scenario_offshore_curtailed_gwh = 0.0

    # Process each area for nuclear
    for area in AREAS:
        logger.info(f"  Processing area: {area}")

        # Get nuclear data if this is a nuclear scenario
        if any(tag in scenario_label for tag in ("N", "SMR", "LMR")):
            try:
                nuke_gen, nuke_revenue, nuke_cf, nuke_vf = get_nuclear_generation_and_revenue(
                    export_df, process_df, area, NUCLEAR_PRICE, scenario_name=scenario_name
                )

                if nuke_gen is not None:
                    n_weather_years = len(nuke_gen.index.get_level_values("scenario").unique())
                    n_hours_per_year = len(nuke_gen) / n_weather_years
                    nuke_generation_gwh = nuke_gen.sum().sum() / n_weather_years / 1000  # MW to GWh
                    nuke_expected_generation_gwh = _expected_nuclear_generation_gwh(
                        capacity_mw=nuclear_capacity_by_area.get(area, 0.0),
                        n_hours_per_year=n_hours_per_year,
                    )
                    nuke_curtailed = max(nuke_expected_generation_gwh - nuke_generation_gwh, 0.0)
                    scenario_nuclear_generation_gwh += nuke_generation_gwh
                    scenario_nuclear_curtailed_gwh += nuke_curtailed
                    nuke_revenue_meur = nuke_revenue / 1e6  # EUR to MEUR

                    # Calculate operating cost
                    nuke_opex_meur = (nuke_generation_gwh * 1000 * NUCLEAR_OPEX) / 1e6  # MWh * EUR/MWh -> MEUR

                    # Calculate net revenue (revenue - opex)
                    nuke_net_revenue_meur = nuke_revenue_meur - nuke_opex_meur

                    # Calculate installed capacity from generation and capacity factor
                    # Generation (MWh) = Capacity (MW) × 8760 × CF
                    # Capacity (MW) = Generation (MWh) / (8760 × CF)
                    nuke_capacity_mw = (nuke_generation_gwh * 1000) / (8760 * nuke_cf) if nuke_cf > 0 else 0.0

                    # Calculate inferred CAPEX for break-even at construction start (FID), EUR/kW.
                    nuke_capacity_kw = nuke_capacity_mw * 1000
                    nuke_capex_fid = _breakeven_capex_fid_per_kw(
                        annual_net_revenue_meur=nuke_net_revenue_meur,
                        capacity_kw=nuke_capacity_kw,
                        annuity_factor=NUCLEAR_ANNUITY_FACTOR,
                        construction_years=NUCLEAR_CONSTRUCTION_YEARS,
                        discount_rate=DISCOUNT_RATE,
                    )

                    results.append(
                        {
                            "scenario": scenario_label,
                            "area": area,
                            "technology": "Nuclear",
                            "generation_gwh": nuke_generation_gwh,
                            "curtailed_gwh": nuke_curtailed,
                            "revenue_meur": nuke_revenue_meur,
                            "opex_meur": nuke_opex_meur,
                            "net_revenue_meur": nuke_net_revenue_meur,
                            "capacity_factor": nuke_cf,
                            "value_factor": nuke_vf,
                            "revenue_per_mwh": nuke_revenue_meur * 1000 / nuke_generation_gwh
                            if nuke_generation_gwh > 0
                            else 0.0,
                            "breakeven_capex_eur_per_kw": nuke_capex_fid,
                            "construction_years": NUCLEAR_CONSTRUCTION_YEARS,
                        }
                    )

                    logger.info(
                        f"    Nuclear: {nuke_generation_gwh:.2f} GWh (curtailed: {nuke_curtailed:.2f} GWh), {nuke_revenue_meur:.2f} M€, CF={nuke_cf:.3f}, VF={nuke_vf:.3f}"
                    )
                else:
                    logger.info(f"    No nuclear in {area}")

            except Exception as e:
                logger.warning(f"    Failed to process nuclear for {area}: {e}")

    if scenario_label.startswith("N-") and scenario_nuclear_generation_gwh > 0:
        target_text = f", target={target_twh:.2f}" if target_twh is not None else ""
        logger.info(
            "  N-case nuclear balance (dispatch + curtailment): "
            f"{(scenario_nuclear_generation_gwh + scenario_nuclear_curtailed_gwh) / 1000:.2f} TWh "
            f"(dispatch={(scenario_nuclear_generation_gwh / 1000):.2f}, "
            f"curtailment={(scenario_nuclear_curtailed_gwh / 1000):.2f}{target_text})"
        )

    # Process offshore wind areas separately
    for area in OFFSHORE_WIND_AREAS:
        logger.info(f"  Processing offshore wind area: {area}")

        # Get offshore wind data if this is an offshore wind scenario
        if "OW" in scenario_label:
            try:
                wind_gen, wind_revenue, wind_cf, wind_vf, wind_curtailed = get_offshore_wind_generation_and_revenue(
                    export_df, process_df, area, scenario_name=scenario_name
                )

                if wind_gen is not None:
                    n_weather_years = len(wind_gen.index.get_level_values("scenario").unique())
                    wind_generation_gwh = wind_gen.sum().sum() / n_weather_years / 1000  # MW to GWh
                    wind_net_generation_gwh = wind_generation_gwh - wind_curtailed  # Subtract curtailment
                    scenario_offshore_generation_gwh += wind_net_generation_gwh
                    scenario_offshore_curtailed_gwh += wind_curtailed
                    wind_revenue_meur = wind_revenue / 1e6  # EUR to MEUR

                    # Calculate operating cost
                    wind_opex_meur = (
                        wind_net_generation_gwh * 1000 * OFFSHORE_WIND_OPEX
                    ) / 1e6  # MWh * EUR/MWh -> MEUR

                    # Calculate net revenue (revenue - opex)
                    wind_net_revenue_meur = wind_revenue_meur - wind_opex_meur

                    # Calculate installed capacity from generation and capacity factor
                    # Generation (MWh) = Capacity (MW) × 8760 × CF
                    wind_capacity_mw = (wind_generation_gwh * 1000) / (8760 * wind_cf) if wind_cf > 0 else 0.0

                    # Calculate inferred CAPEX for break-even at construction start (FID), EUR/kW.
                    wind_capacity_kw = wind_capacity_mw * 1000
                    wind_capex_fid = _breakeven_capex_fid_per_kw(
                        annual_net_revenue_meur=wind_net_revenue_meur,
                        capacity_kw=wind_capacity_kw,
                        annuity_factor=OFFSHORE_WIND_ANNUITY_FACTOR,
                        construction_years=OFFSHORE_WIND_CONSTRUCTION_YEARS,
                        discount_rate=DISCOUNT_RATE,
                    )

                    results.append(
                        {
                            "scenario": scenario_label,
                            "area": area,
                            "technology": "Offshore Wind",
                            "generation_gwh": wind_net_generation_gwh,
                            "curtailed_gwh": wind_curtailed,
                            "gross_generation_gwh": wind_generation_gwh,
                            "revenue_meur": wind_revenue_meur,
                            "opex_meur": wind_opex_meur,
                            "net_revenue_meur": wind_net_revenue_meur,
                            "capacity_factor": wind_cf,
                            "value_factor": wind_vf,
                            "revenue_per_mwh": wind_revenue_meur * 1000 / wind_net_generation_gwh
                            if wind_net_generation_gwh > 0
                            else 0.0,
                            "breakeven_capex_eur_per_kw": wind_capex_fid,
                            "construction_years": OFFSHORE_WIND_CONSTRUCTION_YEARS,
                        }
                    )

                    logger.info(
                        f"    Offshore Wind: {wind_net_generation_gwh:.2f} GWh (curtailed: {wind_curtailed:.2f} GWh), {wind_revenue_meur:.2f} M€, CF={wind_cf:.3f}, VF={wind_vf:.3f}"
                    )
                else:
                    logger.info(f"    No offshore wind in {area}")

            except Exception as e:
                logger.warning(f"    Failed to process offshore wind for {area}: {e}")

    if scenario_label.startswith("OW-") and scenario_offshore_generation_gwh > 0:
        target_text = f", target={target_twh:.2f}" if target_twh is not None else ""
        logger.info(
            "  OW-case offshore balance (generation + spillage): "
            f"{(scenario_offshore_generation_gwh + scenario_offshore_curtailed_gwh) / 1000:.2f} TWh "
            f"(generation={(scenario_offshore_generation_gwh / 1000):.2f}, "
            f"spillage={(scenario_offshore_curtailed_gwh / 1000):.2f}{target_text})"
        )

    if scenario_label.startswith("OWN-") and (
        scenario_nuclear_generation_gwh > 0 or scenario_offshore_generation_gwh > 0
    ):
        target_text = f", target={target_twh:.2f}" if target_twh is not None else ""
        logger.info(
            "  OWN-case total balance (generation + spillage/curtailment): "
            f"{(scenario_nuclear_generation_gwh + scenario_nuclear_curtailed_gwh + scenario_offshore_generation_gwh + scenario_offshore_curtailed_gwh) / 1000:.2f} TWh "
            f"(nuclear={(scenario_nuclear_generation_gwh + scenario_nuclear_curtailed_gwh) / 1000:.2f}, "
            f"offshore={(scenario_offshore_generation_gwh + scenario_offshore_curtailed_gwh) / 1000:.2f}{target_text})"
        )

# Create results DataFrame
df_results = pd.DataFrame(results)

if df_results.empty:
    logger.error("No results generated!")
    raise ValueError("No results generated!")

# Convert GWh to TWh
df_results["generation_twh"] = df_results["generation_gwh"] / 1000

# Save to CSV
output_csv = output_path / "nuclear_offshore_revenue.csv"
df_results.to_csv(output_csv, index=False)
logger.info(f"\nSaved results to {output_csv}")

# Print summary
logger.info("\n" + "=" * 80)
logger.info("SUMMARY BY SCENARIO AND TECHNOLOGY")
logger.info("=" * 80)

summary = (
    df_results.groupby(["scenario", "technology"])
    .agg(
        {
            "generation_twh": "sum",
            "revenue_meur": "sum",
            "capacity_factor": "mean",
            "value_factor": "mean",
        }
    )
    .round(2)
)

print(summary)

# Create LaTeX table
latex_output = output_path / "nuclear_offshore_revenue.tex"

# Prepare data for LaTeX table with areas
latex_rows = []


def _scenario_sort_key(scenario: str) -> tuple[int, int, int, int]:
    # Expected format examples: N-BA+, OWN-LLPS, OW-BA
    parts = scenario.split("-", 1)
    tech = parts[0] if parts else ""
    policy_with_flag = parts[1] if len(parts) > 1 else ""
    has_plus = policy_with_flag.endswith("+")
    policy = policy_with_flag.rstrip("+")

    # Layer order requested by user:
    # 1) "+" or ""
    # 2) BA or LLPS
    # 3) N, OWN, OW
    plus_rank = 0 if has_plus else 1
    policy_rank = {"BA": 0, "LLPS": 1}.get(policy, 99)
    tech_rank = {"N": 0, "OWN": 1, "OW": 2}.get(tech, 99)
    fallback_rank = 0 if (policy_rank != 99 or tech_rank != 99) else 1
    return (fallback_rank, plus_rank, policy_rank, tech_rank)


# Get unique scenarios from results, exclude SMR/LMR from LaTeX table, then sort
scenarios_in_results = sorted(
    [s for s in df_results["scenario"].unique() if not (s.startswith("SMR") or s.startswith("LMR"))],
    key=_scenario_sort_key,
)

for i, scenario in enumerate(scenarios_in_results):
    scenario_data = df_results[df_results["scenario"] == scenario]

    # Collect all rows for this scenario
    scenario_rows = []

    # Nuclear rows by area
    nuclear_data = scenario_data[scenario_data["technology"] == "Nuclear"].sort_values("area")
    for _, row in nuclear_data.iterrows():
        gen = row["generation_twh"]
        if round(gen, 1) == 0.0:
            continue
        rev_per_mwh = row["revenue_per_mwh"]
        curtail = abs(row["curtailed_gwh"]) / 1000  # Convert to TWh and show absolute value
        cf = row["capacity_factor"]
        vf = row["value_factor"]
        capex = row["breakeven_capex_eur_per_kw"]
        area = row["area"]
        scenario_rows.append(
            f"{area} & Nuclear & {gen:.1f} & {rev_per_mwh:.1f} & {curtail:.2f} & {cf:.2f} & {vf:.2f} & {capex:.0f}"
        )

    # Offshore Wind rows by area
    wind_data = scenario_data[scenario_data["technology"] == "Offshore Wind"].sort_values("area")
    for _, row in wind_data.iterrows():
        gen = row["generation_twh"]
        if round(gen, 1) == 0.0:
            continue
        rev_per_mwh = row["revenue_per_mwh"]
        curtail = abs(row["curtailed_gwh"]) / 1000  # Convert to TWh and show absolute value
        cf = row["capacity_factor"]
        vf = row["value_factor"]
        capex = row["breakeven_capex_eur_per_kw"]
        area = row["area"]
        scenario_rows.append(
            f"{area} & Offshore Wind & {gen:.1f} & {rev_per_mwh:.1f} & {curtail:.2f} & {cf:.2f} & {vf:.2f} & {capex:.0f}"
        )

    # Add multirow for scenario
    if scenario_rows:
        n_rows = len(scenario_rows)
        # First row with multirow - format scenario name with \texttt{}
        scenario_formatted = f"\\texttt{{{scenario}}}"
        latex_rows.append(f"\\multirow{{{n_rows}}}{{*}}{{{scenario_formatted}}} & {scenario_rows[0]} \\\\")
        # Remaining rows without scenario name
        for row in scenario_rows[1:]:
            latex_rows.append(f" & {row} \\\\")
        # Add midrule after each scenario except the last
        if i < len(scenarios_in_results) - 1:
            latex_rows.append("\\midrule")

# Write LaTeX table
latex_table = (
    r"""\begin{table}[htbp]
\centering
\caption{Nuclear and Offshore Wind Generation, Revenue, and Performance Factors by Area}
\label{tab:nuclear_offshore_revenue}
\begin{tabular}{lllrrrrrrr}
\toprule
Scenario & Area & Tech. & Gen. & Rev. & Curtail. & Cap. & Value & Inferred CAPEX \\
         &      &       & (TWh) & (EUR/MWh) & (TWh) & Factor & Factor & for break-even (EUR/kW) \\
\midrule
"""
    + "\n".join(latex_rows)
    + r"""
\bottomrule
\end{tabular}
\end{table}"""
)

with open(latex_output, "w") as f:
    f.write(latex_table)

print(latex_table)

logger.info(f"\nSaved LaTeX table to {latex_output}")

logger.info(f"\nAnalysis complete. Results saved to {output_path}")
