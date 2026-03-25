"""
Visualize economic surplus for SMR and LMR scenarios using bar charts.

Creates bar chart visualizations comparing producer, consumer, and societal surplus
across different nuclear reactor configurations (SMR vs LMR).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
OPERATIONAL_COSTS_JSON = Path.cwd() / "data/operational_costs.json"

# Baseline scenario
BASELINE_SCENARIO = "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF"

# SMR and LMR scenarios
SCENARIOS = [
    BASELINE_SCENARIO,
    # SMR scenarios - distributed across all 5 Norwegian areas
    "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR300LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR900LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1200LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    "SMR1600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    # LMR scenarios - concentrated in NO1 and NO2
    "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR2000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR3000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
    "LMR4000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
]

# Scenario labels for display
SCENARIO_LABELS = {
    "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF": "BASELINE",
    "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF": "SMR300_BA",
    "SMR300LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF": "SMR300_LLPS",
    "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF": "SMR600_BA",
    "SMR600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF": "SMR600_LLPS",
    "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF": "SMR900_BA",
    "SMR900LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF": "SMR900_LLPS",
    "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF": "SMR1200_BA",
    "SMR1200LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF": "SMR1200_LLPS",
    "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF": "SMR1600_BA",
    "SMR1600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF": "SMR1600_LLPS",
    "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF": "LMR2000_BA",
    "LMR2000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF": "LMR2000_LLPS",
    "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF": "LMR3000_BA",
    "LMR3000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF": "LMR3000_LLPS",
    "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF": "LMR4000_BA",
    "LMR4000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF": "LMR4000_LLPS",
}

# Nuclear configuration mapping
# SMR: Small Modular Reactors (distributed across all 5 Norwegian areas)
# LMR: Large Modular Reactors (concentrated in NO1 and NO2)
NUCLEAR_CONFIG = {
    "SMR300_BA": "5×300MW SMR",
    "SMR300_LLPS": "5×300MW SMR",
    "SMR600_BA": "5×600MW SMR",
    "SMR600_LLPS": "5×600MW SMR",
    "SMR900_BA": "5×900MW SMR",
    "SMR900_LLPS": "5×900MW SMR",
    "SMR1200_BA": "5×1200MW SMR",
    "SMR1200_LLPS": "5×1200MW SMR",
    "SMR1600_BA": "5×1600MW SMR",
    "SMR1600_LLPS": "5×1600MW SMR",
    "LMR2000_BA": "2×2000MW LMR",
    "LMR2000_LLPS": "2×2000MW LMR",
    "LMR3000_BA": "2×3000MW LMR",
    "LMR3000_LLPS": "2×3000MW LMR",
    "LMR4000_BA": "2×4000MW LMR",
    "LMR4000_LLPS": "2×4000MW LMR",
}

# Norwegian areas
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# All Nordic areas (including offshore wind areas for processing)
ALL_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4", "DK1", "DK2", "FI"]
OFFSHORE_WIND_AREAS = ["SNII", "UN", "VVD"]  # Offshore wind is in separate busbars

# Mapping of offshore wind areas to connected Norwegian areas
CONNECTED_OFFSHORE_AREAS = {
    "SNII": "NO2",
    "UN": "NO2",
    "VVD": "NO5",
}

# Reverse mapping: Norwegian areas to their connected offshore areas
CONNECTED_OFFSHORE_AREAS_REVERSE = {
    "NO2": ["SNII", "UN"],
    "NO5": ["VVD"],
}

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)

logger.info("=" * 80)
logger.info("Starting SMR/LMR Economic Surplus Analysis")
logger.info("=" * 80)
logger.info(f"Model folder: {MODEL_FOLDER}")
logger.info(f"Total scenarios to process: {len(SCENARIOS)}")
logger.info(f"Output directory: {paper_output_path}")

# Load market dispatch outputs
def _load_market_dispatch_data(scenario_path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    processed_dir = scenario_path / "results" / "processed"
    export_file = processed_dir / "market_dispatch.pkl"
    process_file = processed_dir / "market_dispatch.parquet"
    if not export_file.exists() or not process_file.exists():
        logger.warning(f"Missing processed files in {processed_dir}")
        return None
    return pd.read_pickle(export_file), pd.read_parquet(process_file)


scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS}
scenario_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
for name, path in scenario_paths.items():
    data = _load_market_dispatch_data(path)
    if data is None:
        continue
    scenario_data[name] = data

if not scenario_data:
    logger.error("No scenario market dispatch files found")
    exit(1)
logger.info(f"Loaded {len(scenario_data)} scenarios with market dispatch outputs")

# Operational cost lookup by technology
if not OPERATIONAL_COSTS_JSON.exists():
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


reference_prices = {area: 1000.0 for area in ALL_AREAS}
logger.info("Using fixed reference price for CS: 1000 €/MWh")

first_export = next(iter(scenario_data.values()))[0]
n_weather_years = len(first_export.index.get_level_values("scenario").unique())

# Storage for results
surplus_results = {}

logger.info("\n" + "=" * 80)
logger.info("Starting scenario processing")
logger.info("=" * 80)

# Calculate surplus for each scenario
for scenario_name, (export_df, process_df) in scenario_data.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    logger.info(f"\nProcessing scenario: {short_name}")

    try:
        area_surplus = {}

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

            # Consumer Surplus
            ref_price = reference_prices.get(area)
            consumer_surplus = np.sum(load * (ref_price - prices))
            consumer_surplus_meur = consumer_surplus / 1e6

            tech_generation = {
                "hydro": np.nan_to_num(aligned["hydro"].to_numpy(), nan=0.0),
                "solar": np.nan_to_num(aligned["solar"].to_numpy(), nan=0.0),
                "wind_onshore": np.nan_to_num(aligned["onshore_wind"].to_numpy(), nan=0.0),
                "wind_offshore": np.nan_to_num(aligned["offshore_wind"].to_numpy(), nan=0.0),
                "nuclear": np.zeros_like(load),
                "biomass": np.zeros_like(load),
                "fossil_gas": np.zeros_like(load),
                "fossil_other": np.zeros_like(load),
            }

            for tech in ("nuclear", "biomass", "fossil_gas", "fossil_other"):
                if tech in proc_area.columns:
                    tech_generation[tech] = np.nan_to_num(
                        proc_area.reindex(aligned.index)[tech].to_numpy(), nan=0.0
                    )

            # Producer surplus using marginal costs
            producer_surplus = 0.0
            for tech_name, tech_gen in tech_generation.items():
                mc = _operational_cost(tech_name)
                producer_surplus += np.sum(tech_gen * (prices - mc))
            producer_surplus_meur = producer_surplus / 1e6

            # Societal Surplus
            societal_surplus_meur = consumer_surplus_meur + producer_surplus_meur

            area_surplus[area] = {
                "consumer_surplus": consumer_surplus_meur / n_weather_years,
                "producer_surplus": producer_surplus_meur / n_weather_years,
                "societal_surplus": societal_surplus_meur / n_weather_years,
            }

        # Calculate totals (including offshore wind areas as Norwegian-owned)
        norway_and_offshore = NO_AREAS + OFFSHORE_WIND_AREAS
        no_consumer_surplus = sum(area_surplus.get(area, {}).get("consumer_surplus", 0) for area in norway_and_offshore)
        no_producer_surplus = sum(area_surplus.get(area, {}).get("producer_surplus", 0) for area in norway_and_offshore)
        no_societal_surplus = no_consumer_surplus + no_producer_surplus

        total_consumer_surplus = sum(v.get("consumer_surplus", 0) for v in area_surplus.values())
        total_producer_surplus = sum(v.get("producer_surplus", 0) for v in area_surplus.values())
        total_societal_surplus = total_consumer_surplus + total_producer_surplus

        surplus_results[short_name] = {
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

        logger.info(
            f"  Norway: CS={no_consumer_surplus:.1f} M€, PS={no_producer_surplus:.1f} M€, SS={no_societal_surplus:.1f} M€"
        )

    except Exception as e:
        logger.error(f"Failed to process {scenario_name}: {e}")
        import traceback

        traceback.print_exc()

# ============================================================================
# Create visualizations
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("Creating visualizations")
logger.info("=" * 80)

# Define colors
COLORS = {
    "consumer": "#1f77b4",  # Blue
    "producer": "#ff7f0e",  # Orange
    "societal": "#2ca02c",  # Green
}

# Define colors for reactor types
REACTOR_COLORS = {
    "SMR": "#1f77b4",  # Blue
    "LMR": "#d62728",  # Red
}

# Prepare data for plotting - separate SMR and LMR, and BA vs LLPS
smr_ba_scenarios = [s for s in SCENARIO_LABELS.values() if "SMR" in s and "_BA" in s]
smr_llps_scenarios = [s for s in SCENARIO_LABELS.values() if "SMR" in s and "_LLPS" in s]
lmr_ba_scenarios = [s for s in SCENARIO_LABELS.values() if "LMR" in s and "_BA" in s]
lmr_llps_scenarios = [s for s in SCENARIO_LABELS.values() if "LMR" in s and "_LLPS" in s]

# Sort scenarios by capacity
smr_ba_scenarios.sort(key=lambda x: int(x.split("_")[0][3:]))
smr_llps_scenarios.sort(key=lambda x: int(x.split("_")[0][3:]))
lmr_ba_scenarios.sort(key=lambda x: int(x.split("_")[0][3:]))
lmr_llps_scenarios.sort(key=lambda x: int(x.split("_")[0][3:]))


# Extract nuclear capacity from scenario names
def get_nuclear_capacity(scenario_name):
    """Extract total nuclear capacity in MW from scenario name."""
    if scenario_name == "BASELINE":
        return 0
    if "SMR" in scenario_name:
        # SMR: 5 reactors, capacity per reactor
        capacity_per_reactor = int(scenario_name.split("_")[0][3:])
        return 5 * capacity_per_reactor
    elif "LMR" in scenario_name:
        # LMR: 2 reactors, capacity per reactor
        capacity_per_reactor = int(scenario_name.split("_")[0][3:])
        return 2 * capacity_per_reactor
    return 0


# Get baseline values
baseline_no = surplus_results["BASELINE"]["norway_total"]
baseline_all = surplus_results["BASELINE"]["all_areas_total"]

# ============================================================================
# Figure 1: Scatter plot - Nuclear Capacity vs Economic Surplus (Norway)
# ============================================================================
logger.info("\nCreating scatter plot: Nuclear Capacity vs Economic Surplus (Norway)")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Collect data for all scenarios (BA and LLPS)
all_scenarios = smr_ba_scenarios + smr_llps_scenarios + lmr_ba_scenarios + lmr_llps_scenarios

# Separate by load profile
ba_scenarios = [s for s in all_scenarios if "_BA" in s]
llps_scenarios = [s for s in all_scenarios if "_LLPS" in s]

# Define line styles for different surplus types
line_styles = {
    "societal": "-",  # Solid line for Societal Surplus
    "consumer": "--",  # Dashed line for Consumer Surplus
    "producer": "-.",  # Dash-dot line for Producer Surplus
}

# Plot both SMR and LMR
for reactor_type in ["SMR", "LMR"]:
    for scenario_list, marker, load_label in [(ba_scenarios, "o", "BA"), (llps_scenarios, "s", "LLPS")]:
        reactor_list = [s for s in scenario_list if reactor_type in s]

        for surplus_type, linestyle in line_styles.items():
            capacities = []
            surplus_changes = []
            for s in reactor_list:
                if s in surplus_results:
                    cap = get_nuclear_capacity(s)
                    no_total = surplus_results[s]["norway_total"]
                    if surplus_type == "consumer":
                        delta = (no_total["consumer_surplus"] - baseline_no["consumer_surplus"]) / 1000
                    elif surplus_type == "producer":
                        delta = (no_total["producer_surplus"] - baseline_no["producer_surplus"]) / 1000
                    else:  # societal
                        delta = (no_total["societal_surplus"] - baseline_no["societal_surplus"]) / 1000
                    capacities.append(cap)
                    surplus_changes.append(delta)

            if capacities:
                # Sort by capacity for proper line plotting
                sorted_pairs = sorted(zip(capacities, surplus_changes))
                capacities_sorted = [x[0] for x in sorted_pairs]
                surplus_sorted = [x[1] for x in sorted_pairs]

                ax.plot(
                    capacities_sorted,
                    surplus_sorted,
                    linestyle,
                    color=REACTOR_COLORS[reactor_type],
                    alpha=0.5,
                    linewidth=2,
                    label=f"{reactor_type} {surplus_type.capitalize()} {load_label}",
                )
                ax.scatter(
                    capacities_sorted,
                    surplus_sorted,
                    s=100,
                    alpha=0.6,
                    marker=marker,
                    color=REACTOR_COLORS[reactor_type],
                )

ax.set_xlabel("Nuclear Capacity Added (MW)", fontsize=12, fontweight="bold")
ax.set_ylabel("Δ Economic Surplus (Billion €)", fontsize=12, fontweight="bold")
ax.set_title(
    "Economic Surplus vs Nuclear Capacity - Norway\nSMR (Distributed 5×SMR) vs LMR (Concentrated 2×LMR)",
    fontsize=13,
    fontweight="bold",
)

# Create custom legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Line style legend
line_legend = [
    Line2D([0], [0], color="gray", linestyle="-", linewidth=2, label="Societal Surplus"),
    Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Consumer Surplus"),
    Line2D([0], [0], color="gray", linestyle="-.", linewidth=2, label="Producer Surplus"),
]

# Reactor type legend
reactor_legend = [
    Patch(facecolor=REACTOR_COLORS["SMR"], label="SMR (blue)"),
    Patch(facecolor=REACTOR_COLORS["LMR"], label="LMR (red)"),
]

# Marker legend
marker_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="BA Load Profile"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="LLPS Load Profile"),
]

# Combine legends
first_legend = ax.legend(handles=line_legend, loc="upper left", fontsize=9, title="Surplus Type", framealpha=0.9)
ax.add_artist(first_legend)

second_legend = ax.legend(handles=reactor_legend, loc="upper right", fontsize=9, title="Reactor Type", framealpha=0.9)
ax.add_artist(second_legend)

third_legend = ax.legend(handles=marker_legend, loc="lower right", fontsize=9, title="Load Profile", framealpha=0.9)

ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

plt.suptitle(
    "Economic Surplus Change from Baseline\nCircles: BA | Squares: LLPS | Blue: SMR | Red: LMR",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_file_scatter_no = paper_output_path / "nuclear_capacity_vs_surplus_norway.pdf"
plt.savefig(output_file_scatter_no, dpi=300, bbox_inches="tight")
logger.info(f"\nSaved nuclear capacity scatter plot (Norway) to: {output_file_scatter_no}")

# ============================================================================
# Figure 2: Scatter plot - Nuclear Capacity vs Economic Surplus (All Nordic)
# ============================================================================
logger.info("\nCreating scatter plot: Nuclear Capacity vs Economic Surplus (All Nordic)")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Define line styles for different surplus types
line_styles = {
    "societal": "-",  # Solid line for Societal Surplus
    "consumer": "--",  # Dashed line for Consumer Surplus
    "producer": "-.",  # Dash-dot line for Producer Surplus
}

# Plot both SMR and LMR
for reactor_type in ["SMR", "LMR"]:
    for scenario_list, marker, load_label in [(ba_scenarios, "o", "BA"), (llps_scenarios, "s", "LLPS")]:
        reactor_list = [s for s in scenario_list if reactor_type in s]

        for surplus_type, linestyle in line_styles.items():
            capacities = []
            surplus_changes = []
            for s in reactor_list:
                if s in surplus_results:
                    cap = get_nuclear_capacity(s)
                    all_total = surplus_results[s]["all_areas_total"]
                    if surplus_type == "consumer":
                        delta = (all_total["consumer_surplus"] - baseline_all["consumer_surplus"]) / 1000
                    elif surplus_type == "producer":
                        delta = (all_total["producer_surplus"] - baseline_all["producer_surplus"]) / 1000
                    else:  # societal
                        delta = (all_total["societal_surplus"] - baseline_all["societal_surplus"]) / 1000
                    capacities.append(cap)
                    surplus_changes.append(delta)

            if capacities:
                # Sort by capacity for proper line plotting
                sorted_pairs = sorted(zip(capacities, surplus_changes))
                capacities_sorted = [x[0] for x in sorted_pairs]
                surplus_sorted = [x[1] for x in sorted_pairs]

                ax.plot(
                    capacities_sorted,
                    surplus_sorted,
                    linestyle,
                    color=REACTOR_COLORS[reactor_type],
                    alpha=0.5,
                    linewidth=2,
                    label=f"{reactor_type} {surplus_type.capitalize()} {load_label}",
                )
                ax.scatter(
                    capacities_sorted,
                    surplus_sorted,
                    s=100,
                    alpha=0.6,
                    marker=marker,
                    color=REACTOR_COLORS[reactor_type],
                )

ax.set_xlabel("Nuclear Capacity Added (MW)", fontsize=12, fontweight="bold")
ax.set_ylabel("Δ Economic Surplus (Billion €)", fontsize=12, fontweight="bold")
ax.set_title(
    "Economic Surplus vs Nuclear Capacity - All Nordic\nSMR (Distributed 5×SMR) vs LMR (Concentrated 2×LMR)",
    fontsize=13,
    fontweight="bold",
)

# Create custom legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Line style legend
line_legend = [
    Line2D([0], [0], color="gray", linestyle="-", linewidth=2, label="Societal Surplus"),
    Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Consumer Surplus"),
    Line2D([0], [0], color="gray", linestyle="-.", linewidth=2, label="Producer Surplus"),
]

# Reactor type legend
reactor_legend = [
    Patch(facecolor=REACTOR_COLORS["SMR"], label="SMR (blue)"),
    Patch(facecolor=REACTOR_COLORS["LMR"], label="LMR (red)"),
]

# Marker legend
marker_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="BA Load Profile"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="LLPS Load Profile"),
]

# Combine legends
first_legend = ax.legend(handles=line_legend, loc="upper left", fontsize=9, title="Surplus Type", framealpha=0.9)
ax.add_artist(first_legend)

second_legend = ax.legend(handles=reactor_legend, loc="upper right", fontsize=9, title="Reactor Type", framealpha=0.9)
ax.add_artist(second_legend)

third_legend = ax.legend(handles=marker_legend, loc="lower right", fontsize=9, title="Load Profile", framealpha=0.9)

ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

plt.suptitle(
    "Economic Surplus Change from Baseline\nCircles: BA | Squares: LLPS | Blue: SMR | Red: LMR",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_file_scatter_all = paper_output_path / "nuclear_capacity_vs_surplus_nordic.pdf"
plt.savefig(output_file_scatter_all, dpi=300, bbox_inches="tight")
logger.info(f"Saved nuclear capacity scatter plot (Nordic) to: {output_file_scatter_all}")


# Create summary CSV
logger.info("\nCreating summary CSV...")

summary_data = []
all_scenarios = ["BASELINE"] + smr_ba_scenarios + smr_llps_scenarios + lmr_ba_scenarios + lmr_llps_scenarios

for scenario_name in all_scenarios:
    if scenario_name in surplus_results:
        no_total = surplus_results[scenario_name]["norway_total"]
        all_total = surplus_results[scenario_name]["all_areas_total"]

        if scenario_name == "BASELINE":
            reactor_type = "None"
            load_profile = "BA"
            nuclear_config = "N/A"
        else:
            nuclear_config = NUCLEAR_CONFIG.get(scenario_name, "N/A")
            reactor_type = "SMR" if "SMR" in scenario_name else "LMR"
            load_profile = scenario_name.split("_")[1]

        summary_data.append(
            {
                "Scenario": scenario_name,
                "Reactor_Type": reactor_type,
                "Load_Profile": load_profile,
                "Nuclear_Config": nuclear_config,
                "Region": "Norway",
                "Consumer_Surplus_BEur": no_total["consumer_surplus"] / 1000,
                "Producer_Surplus_BEur": no_total["producer_surplus"] / 1000,
                "Societal_Surplus_BEur": no_total["societal_surplus"] / 1000,
            }
        )

        summary_data.append(
            {
                "Scenario": scenario_name,
                "Reactor_Type": reactor_type,
                "Load_Profile": load_profile,
                "Nuclear_Config": nuclear_config,
                "Region": "All_Nordic",
                "Consumer_Surplus_BEur": all_total["consumer_surplus"] / 1000,
                "Producer_Surplus_BEur": all_total["producer_surplus"] / 1000,
                "Societal_Surplus_BEur": all_total["societal_surplus"] / 1000,
            }
        )

    df_summary = pd.DataFrame(summary_data)
    output_csv = paper_output_path / "smr_lmr_surplus_summary.csv"
    df_summary.to_csv(output_csv, index=False, float_format="%.2f")
    logger.info(f"  ✓ Saved: {output_csv.name}")

    print("\n" + "=" * 100)
    print("SMR/LMR SURPLUS SUMMARY")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)

logger.info("\n" + "=" * 80)
logger.info("✓ Script completed successfully!")
logger.info("=" * 80)
logger.info("Generated files:")
try:
    logger.info(f"  - {output_file_scatter_no.name}")
    logger.info(f"  - {output_file_scatter_all.name}")
    logger.info(f"  - {output_csv.name}")
except NameError as e:
    logger.warning(f"Some output files may not have been created: {e}")
logger.info("=" * 80)
