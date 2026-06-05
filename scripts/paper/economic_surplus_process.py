"""
Calculate producer surplus, consumer surplus, and societal surplus for OW, N, OWN scenarios.

Economic surplus definitions:
- Consumer Surplus: Benefit to consumers from purchasing at market price vs. willingness to pay
  Approximated as: 0.5 * quantity * (reference_price - market_price)

- Producer Surplus: Revenue minus production costs
  Calculated as: sum(generation * (price - marginal_cost))

- Societal Surplus (Total Welfare): Consumer Surplus + Producer Surplus
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import logger
from scripts.paper.processed_dispatch import load_processed_dispatch_data

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
OPERATIONAL_COSTS_JSON = Path.cwd() / "data/operational_costs.json"

# Scenarios from OW_N_OWN group
SCENARIOS = [
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
]

# Shorter names for display
SCENARIO_LABELS = {
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF": "B+",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-LLPS+",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-LLPS+",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-LLPS+",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA+",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-BA+",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA+",
}

# Group scenarios by type
SCENARIO_GROUPS = {
    "N": ["N-LLPS+", "N-BA+"],
    "OWN": ["OWN-LLPS+", "OWN-BA+"],
    "OW": ["OW-LLPS+", "OW-BA+"],
}

# Norwegian areas
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# All Nordic areas (including offshore wind areas for processing)
ALL_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4", "DK1", "DK2", "FI"]
OFFSHORE_WIND_AREAS = ["SNII", "UN", "VVD"]  # Offshore wind is in separate busbars
CONNECTED_OFFSHORE_AREAS = {
    "SNII": "NO2",
    "UN": "NO2",
    "VVD": "NO5",
}

CONNECTED_OFFSHORE_AREAS_REVERSE = {
    "NO2": ["SNII", "UN"],
    "NO5": ["VVD"],
}

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)


def _load_processed_dispatch_data(scenario_path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    data = load_processed_dispatch_data(scenario_path, require_market_step_technologies=True)
    if data is None:
        logger.warning(f"Missing processed_data.parquet dispatch data for {scenario_path.name}")
    return data


# Load scenarios from processed_data.parquet
scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS}
scenario_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
for name, path in scenario_paths.items():
    data = _load_processed_dispatch_data(path)
    if data is None:
        continue
    scenario_data[name] = data

if not scenario_data:
    logger.error("No scenario processed dispatch data found")
    exit(1)

logger.info(f"Loaded {len(scenario_data)} scenarios from processed_data.parquet")

# Operational cost lookup by technology
if not OPERATIONAL_COSTS_JSON.exists():
    # Backward-compatible fallback to current repository filename
    OPERATIONAL_COSTS_JSON = Path.cwd() / "data/operational_cost.json"

if not OPERATIONAL_COSTS_JSON.exists():
    raise FileNotFoundError(
        f"Operational cost file not found: {Path.cwd() / 'data/operational_costs.json'} "
        f"(or fallback {Path.cwd() / 'data/operational_cost.json'})"
    )

with open(OPERATIONAL_COSTS_JSON) as f:
    op_cost_raw = pd.read_json(f, typ="series")

op_cost_by_tech = {}
if all(not isinstance(v, (dict, list)) for v in op_cost_raw.values):
    for key, val in op_cost_raw.items():
        if pd.isna(val):
            raise ValueError(f"Operational cost is NaN for technology '{key}' in {OPERATIONAL_COSTS_JSON}")
        op_cost_by_tech[str(key).strip().lower()] = float(val)
else:
    op_cost_payload = pd.json_normalize(op_cost_raw["technologies"])
    for _, row in op_cost_payload.iterrows():
        tech_name = str(row["technology"]).strip().lower()
        cost_val = row.get("operational_cost")
        if pd.isna(cost_val):
            raise ValueError(f"Operational cost is NaN/missing for technology '{tech_name}' in {OPERATIONAL_COSTS_JSON}")
        op_cost_by_tech[tech_name] = float(cost_val)

tech_cost_aliases = {
    "hydro": ["hydro"],
    "solar": ["solar"],
    "wind_onshore": ["wind_onshore", "onshore_wind", "wind onshore"],
    "wind_offshore": ["wind_offshore", "offshore_wind", "wind offshore"],
    "nuclear": ["nuclear", "nuclear (new)"],
    "biomass": ["biomass"],
    "fossil_gas": ["fossil_gas", "fossil gas"],
    "fossil_other": ["fossil_other", "fossil other"],
}


def _operational_cost(tech: str) -> float:
    for candidate in tech_cost_aliases.get(tech, [tech]):
        if candidate in op_cost_by_tech:
            return float(op_cost_by_tech[candidate])
    available = ", ".join(sorted(op_cost_by_tech.keys()))
    raise KeyError(
        f"Missing operational cost for tech '{tech}'. Tried aliases {tech_cost_aliases.get(tech, [tech])}. "
        f"Available keys in {OPERATIONAL_COSTS_JSON}: {available}"
    )


logger.info(f"Using operational cost file: {OPERATIONAL_COSTS_JSON}")
logger.info(
    "Operational cost resolved for onshore wind: "
    f"{_operational_cost('wind_onshore'):.4f} €/MWh"
)


# For consumer surplus calculation, we need a reference price
# Using baseline average price as reference (willingness to pay proxy)
# This assumes demand is relatively inelastic in the short run
reference_prices = {area: 1000.0 for area in ALL_AREAS}
logger.info("Using fixed reference price for CS: 1000 €/MWh")

n_weather_years = 30

# Storage for results
surplus_results = {}
technology_surplus_results = {}

# Calculate surplus for each scenario
for scenario_name, (export_df, process_df) in scenario_data.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    logger.info(f"\nProcessing scenario: {short_name}")

    try:
        # Initialize storage for this scenario
        area_surplus = {}
        tech_surplus = {
            "hydro": 0,
            "solar": 0,
            "wind_onshore": 0,
            "wind_offshore": 0,
            "nuclear": 0,
            "biomass": 0,
            "fossil_gas": 0,
            "fossil_other": 0,
        }

        for area in ALL_AREAS:
            if area not in export_df.index.get_level_values("area"):
                continue
            if area not in process_df.index.get_level_values("area"):
                continue

            exp_area = export_df.xs(area, level="area")
            proc_area = process_df.xs(area, level="area")
            aligned = exp_area.join(proc_area[["market_price"]], how="inner")
            if aligned.empty:
                continue

            prices = np.nan_to_num(aligned["market_price"].to_numpy(), nan=0.0)
            load = np.nan_to_num(aligned["load"].to_numpy(), nan=0.0)

            # Skip areas with no load (offshore wind areas)
            if np.sum(load) == 0:
                # For generation-only areas (offshore), we'll process generation but skip consumer surplus
                # Use mean price for weighting since there's no load
                avg_price = np.mean(prices)
            else:
                avg_price = np.average(prices, weights=load)

            # Consumer Surplus Calculation
            # CS = sum(load * (reference_price - market_price))
            # Using baseline price as proxy for maximum willingness to pay
            # This represents the benefit consumers get from paying market price instead of reference
            ref_price = reference_prices.get(area, prices.mean())

            # Consumer surplus: benefit from paying less than reference price
            consumer_surplus = np.sum(load * (ref_price - prices))

            # Convert from €*MW*h to M€ (million euros)
            consumer_surplus_meur = consumer_surplus / 1e6

            # Producer Surplus Calculation
            # PS = Revenue - Opertional cost

            # Get all generation sources from processed dispatch data
            total_generation = np.zeros_like(load)
            tech_generation = {
                "hydro": np.zeros_like(load),
                "solar": np.zeros_like(load),
                "wind_onshore": np.zeros_like(load),
                "wind_offshore": np.zeros_like(load),
                "nuclear": np.zeros_like(load),
                "biomass": np.zeros_like(load),
                "fossil_gas": np.zeros_like(load),
                "fossil_other": np.zeros_like(load),
            }

            # Hydro generation
            hydro_gen = np.nan_to_num(aligned["hydro"].to_numpy(), nan=0.0)
            total_generation += hydro_gen
            tech_generation["hydro"] += hydro_gen

            # Market steps disaggregated by technology (from process output)
            for tech in ("nuclear", "biomass", "fossil_gas", "fossil_other"):
                if tech in proc_area.columns:
                    tech_gen = np.nan_to_num(proc_area.reindex(aligned.index)[tech].to_numpy(), nan=0.0)
                    total_generation += tech_gen
                    tech_generation[tech] += tech_gen

            # Solar
            solar_gen = np.nan_to_num(aligned["solar"].to_numpy(), nan=0.0)
            total_generation += solar_gen
            tech_generation["solar"] += solar_gen

            # Onshore wind
            wind_on_gen = np.nan_to_num(aligned["onshore_wind"].to_numpy(), nan=0.0)
            total_generation += wind_on_gen
            tech_generation["wind_onshore"] += wind_on_gen

            # Offshore wind
            wind_off_gen = np.nan_to_num(aligned["offshore_wind"].to_numpy(), nan=0.0)
            total_generation += wind_off_gen
            tech_generation["wind_offshore"] += wind_off_gen

            # Add generation from potentially connected offshore areas
            if area in CONNECTED_OFFSHORE_AREAS_REVERSE:
                for offshore_area in CONNECTED_OFFSHORE_AREAS_REVERSE[area]:
                    if offshore_area in export_df.index.get_level_values("area"):
                        exp_off = export_df.xs(offshore_area, level="area").reindex(aligned.index, fill_value=0.0)
                        off_wind = np.nan_to_num(exp_off["offshore_wind"].to_numpy(), nan=0.0)
                        total_generation += off_wind
                        tech_generation["wind_offshore"] += off_wind
                        # Include offshore market steps in disaggregated techs if present
                        if offshore_area in process_df.index.get_level_values("area"):
                            proc_off = process_df.xs(offshore_area, level="area").reindex(aligned.index, fill_value=0.0)
                            for tech in ("nuclear", "biomass", "fossil_gas", "fossil_other"):
                                if tech in proc_off.columns:
                                    tech_gen = np.nan_to_num(proc_off[tech].to_numpy(), nan=0.0)
                                    total_generation += tech_gen
                                    tech_generation[tech] += tech_gen

            # Producer surplus (price minus marginal cost)
            producer_surplus = 0.0
            for tech_name, tech_gen in tech_generation.items():
                mc = _operational_cost(tech_name)
                producer_surplus += np.sum(tech_gen * (prices - mc))

            # Convert to M€
            producer_surplus_meur = producer_surplus / 1e6

            # Calculate producer surplus by technology
            for tech_name, tech_gen in tech_generation.items():
                mc = _operational_cost(tech_name)
                tech_ps = np.sum(tech_gen * (prices - mc)) / 1e6
                # Accumulate for Norwegian areas + offshore wind areas (since they're Norwegian-owned)
                if area in NO_AREAS or area in OFFSHORE_WIND_AREAS:
                    tech_surplus[tech_name] += tech_ps / n_weather_years

            # Societal Surplus (Total Welfare)
            societal_surplus_meur = consumer_surplus_meur + producer_surplus_meur

            # Store results for this area (divide by n_weather_years for expected values)
            area_surplus[area] = {
                "consumer_surplus": consumer_surplus_meur / n_weather_years,
                "producer_surplus": producer_surplus_meur / n_weather_years,
                "societal_surplus": societal_surplus_meur / n_weather_years,
                "avg_price": avg_price,
                "total_load": np.sum(load),
            }

            logger.info(f"  {area}:")
            logger.info(f"    Consumer Surplus: {consumer_surplus_meur / n_weather_years:.1f} M€ (expected)")
            logger.info(f"    Producer Surplus: {producer_surplus_meur / n_weather_years:.1f} M€ (expected)")
            logger.info(f"    Societal Surplus: {societal_surplus_meur / n_weather_years:.1f} M€ (expected)")

        # Calculate total across all Norwegian areas (including offshore wind)
        norway_and_offshore = NO_AREAS + OFFSHORE_WIND_AREAS
        no_consumer_surplus = sum(area_surplus.get(area, {}).get("consumer_surplus", 0) for area in norway_and_offshore)
        no_producer_surplus = sum(area_surplus.get(area, {}).get("producer_surplus", 0) for area in norway_and_offshore)
        no_societal_surplus = no_consumer_surplus + no_producer_surplus

        # Calculate total across all areas
        total_consumer_surplus = sum(v.get("consumer_surplus", 0) for v in area_surplus.values())
        total_producer_surplus = sum(v.get("producer_surplus", 0) for v in area_surplus.values())
        total_societal_surplus = total_consumer_surplus + total_producer_surplus

        surplus_results[short_name] = {
            "by_area": area_surplus,
            "norway_total": {
                "consumer_surplus": no_consumer_surplus,
                "producer_surplus": no_producer_surplus,
                "societal_surplus": no_societal_surplus,
            },
            "all_areas_total": {
                "consumer_surplus": total_consumer_surplus,
                "producer_surplus": total_producer_surplus,
                "societal_surplus": total_societal_surplus,
            },
        }

        # Store technology-level surplus for Norway
        technology_surplus_results[short_name] = tech_surplus

        logger.info("\n  Norway Total:")
        logger.info(f"    Consumer Surplus: {no_consumer_surplus:.1f} M€")
        logger.info(f"    Producer Surplus: {no_producer_surplus:.1f} M€")
        logger.info(f"    Societal Surplus: {no_societal_surplus:.1f} M€")

        logger.info("\n  All Areas Total:")
        logger.info(f"    Consumer Surplus: {total_consumer_surplus:.1f} M€")
        logger.info(f"    Producer Surplus: {total_producer_surplus:.1f} M€")
        logger.info(f"    Societal Surplus: {total_societal_surplus:.1f} M€")

    except Exception as e:
        logger.error(f"Failed to process {scenario_name}: {e}")
        import traceback

        traceback.print_exc()

processed_output_path = paper_output_path / "economic_surplus_data.pkl"
payload = {
    "surplus_results": surplus_results,
    "technology_surplus_results": technology_surplus_results,
    "scenario_labels": SCENARIO_LABELS,
    "scenario_groups": SCENARIO_GROUPS,
    "all_areas": ALL_AREAS,
    "no_areas": NO_AREAS,
    "offshore_areas": OFFSHORE_WIND_AREAS,
}
pd.to_pickle(payload, processed_output_path)
logger.info(f"Saved processed surplus data to: {processed_output_path}")
