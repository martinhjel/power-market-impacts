"""
Calculate value factor and additional revenue for uprated hydropower plants.

Analyzes the economic benefits of hydropower capacity uprating across OW, OWN, and N scenarios.
Computes:
- Value factor: ratio of revenue per MWh to average market price
- Additional annual revenue from uprating
- Revenue per MW of added capacity
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd() / "data"))
from uprate_hydro import uprate_values as UPRATED_PLANTS  # type: ignore

from scripts.common import load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Scenarios from OW_N_OWN group - UPRATED
SCENARIOS_UPRATED = [
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
]

# Non-uprated scenarios for comparison
SCENARIOS_NO_UPRATE = [
    "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
    "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
]

SCENARIOS = SCENARIOS_UPRATED + SCENARIOS_NO_UPRATE

# Baseline without uprating for comparison
BASELINE_NO_UPRATE = "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF"

# Shorter names for display
SCENARIO_LABELS = {
    # Uprated scenarios (+)
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF": "B+",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-LLPS+",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-LLPS+",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-LLPS+",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA+",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-BA+",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA+",

    # Non-uprated scenarios
    "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF": "B",
    "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-LLPS",
    "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-LLPS",
    "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-LLPS",
    "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA",
    "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-BA",
    "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA",

    # Old baseline for comparison
    "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF": "B30",
}

# Map non-uprated short labels to their uprated counterpart for line overlays
NO_UPRATE_TO_UPRATE_LABEL = {
    "B": "B+",
    "N-LLPS": "N-LLPS+",
    "OWN-LLPS": "OWN-LLPS+",
    "OW-LLPS": "OW-LLPS+",
    "N-BA": "N-BA+",
    "OWN-BA": "OWN-BA+",
    "OW-BA": "OW-BA+",
}

# Norwegian areas
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)

# Load scenarios
logger.info("Loading scenarios...")
scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS + [BASELINE_NO_UPRATE]}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

logger.info(f"Loaded {len(scenarios)} scenarios")

# Import after loading scenarios
import json

from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

# ============================================================================
# Load dataset configurations to get plant capacities
# ============================================================================

logger.info("\nLoading dataset configurations to extract plant capacities...")

# Load the uprated baseline configuration
uprate_config_path = (
    base_path
    / f"ltm_output/{MODEL_FOLDER}/BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF/run_folder/emps/ltm_model.json"
)
no_uprate_config_path = (
    base_path
    / f"ltm_output/{MODEL_FOLDER}/BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF/run_folder/emps/ltm_model.json"
)

# Load JSON directly to access plant data
with open(uprate_config_path, "r") as f:
    uprate_config_json = json.load(f)
with open(no_uprate_config_path, "r") as f:
    no_uprate_config_json = json.load(f)

# Extract plant lists
uprate_plants_data = uprate_config_json["model"]["plants"]
no_uprate_plants_data = no_uprate_config_json["model"]["plants"]

# Create lookup dictionaries
uprate_plants_dict = {p["name"]: p for p in uprate_plants_data}
no_uprate_plants_dict = {p["name"]: p for p in no_uprate_plants_data}
logger.info(f"Loaded {len(uprate_plants_dict)} plants from uprate config")
logger.info(f"Loaded {len(no_uprate_plants_dict)} plants from no-uprate config")

# Extract plant capacities and calculate uprated capacity by area
uprated_plant_info = {}
total_added_capacity_by_area = {area: 0 for area in NO_AREAS}

for plant_name in UPRATED_PLANTS:
    # Find plant in uprated config (plants are named "plant_<name>" in the config)
    config_plant_name = f"plant_{plant_name}"

    logger.info(f"Looking for plant: {config_plant_name}")

    if config_plant_name in uprate_plants_dict and config_plant_name in no_uprate_plants_dict:
        uprate_plant = uprate_plants_dict[config_plant_name]
        no_uprate_plant = no_uprate_plants_dict[config_plant_name]

        logger.info(f"  Found {config_plant_name} in both configs")

        # Get capacity from PQ curves - access the first timestamp's y values
        pq_curves_uprate = uprate_plant.get("pq_curves", {})
        pq_curves_no_uprate = no_uprate_plant.get("pq_curves", {})

        logger.info(f"  PQ curves uprates: {list(pq_curves_uprate.values())}")
        logger.info(f"  PQ curves no-uprate: {list(pq_curves_no_uprate.values())}")

        if pq_curves_uprate and pq_curves_no_uprate:
            # Get the first timestamp key (should be '2024-01-01T00:00:00Z')
            timestamp_key = list(pq_curves_uprate.keys())[0]

            capacity_uprate = pq_curves_uprate[timestamp_key]["y"][-1]  # Max power
            capacity_no_uprate = pq_curves_no_uprate.get(timestamp_key, {}).get("y", [0])[-1]

            added_capacity = capacity_uprate - capacity_no_uprate

            logger.info(
                f"  Capacity: {capacity_no_uprate:.1f} MW → {capacity_uprate:.1f} MW (diff: {added_capacity:.1f} MW)"
            )

            # Find the area for this plant via metadata
            plant_area = UPRATED_PLANTS[plant_name].get("elspot_area")
            logger.info(f"  Elspot area: {plant_area} for plant '{plant_name}'")

            if plant_area and added_capacity > 0:
                uprated_plant_info[plant_name] = {
                    "area": plant_area,
                    "capacity_uprate_mw": capacity_uprate,
                    "capacity_no_uprate_mw": capacity_no_uprate,
                    "added_capacity_mw": added_capacity,
                }
                total_added_capacity_by_area[plant_area] += added_capacity
                logger.info(
                    f"  {plant_name} ({plant_area}): {capacity_no_uprate:.1f} MW → {capacity_uprate:.1f} MW (+{added_capacity:.1f} MW)"
                )

logger.info("\nTotal added capacity by area:")
for area in NO_AREAS:
    if total_added_capacity_by_area[area] > 0:
        logger.info(f"  {area}: {total_added_capacity_by_area[area]:.1f} MW")

total_added_capacity_mw = sum(total_added_capacity_by_area.values())
logger.info(f"\nTotal Norway: {total_added_capacity_mw:.1f} MW")

# ============================================================================
# Calculate hydro generation and revenue for each scenario
# ============================================================================

logger.info("\nCalculating hydropower generation and revenue...")

area_hydro_results = {}
uprate_hydro_results = {}
for scenario_name, scenario in scenarios.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    logger.info(f"\nProcessing scenario: {short_name}")

    busbars_dict = scenario.get_busbars()

    # Initialize storage for this scenario
    scenario_area_data = {}
    scenario_uprate_data = {}

    power_prices = {}
    loads = {}
    res_found = defaultdict()
    res_generations = defaultdict(dict)
    hydro_generations = defaultdict(pd.DataFrame)

    for area in NO_AREAS:
        if area not in busbars_dict:
            continue

        busbar = busbars_dict[area]

        # get hydropower generation for uprated plants
        for p in UPRATED_PLANTS:
            reservoir_name = UPRATED_PLANTS[p]["reservoirs"][0]
            # print(reservoir_name)
            reservoirs = busbar.reservoirs()

            res_f = [r for r in reservoirs if f"res_{reservoir_name.lower()}" == r.name.lower()]
            if res_f:
                # Keep plant id as key so downstream joins with uprated_plant_info work.
                res_found[p] = {}
                res_found[p]["res"] = res_f
                res_found[p]["area"] = area
                res_found[p]["reservoir"] = reservoir_name

        max_cap = {}
        for plant_id, plant_data in res_found.items():
            df_prod = df_from_pyltm_result(plant_data["res"][0].production())
            max_cap[plant_id] = df_prod.max().max()
            res_generations[plant_id]["gen"] = df_prod
            res_generations[plant_id]["area"] = plant_data["area"]

        # Get prices
        df_price = df_from_pyltm_result(busbar.market_result_price())
        power_prices[area] = df_price

        # Get hydro generation
        df_hydro = df_from_pyltm_result(busbar.sum_hydro_production())
        hydro_generations[area] = df_hydro

        # Get load in area
        load = df_from_pyltm_result(busbar.sum_load())
        loads[area] = load

    for area in NO_AREAS:
        hydro = hydro_generations[area].values.flatten()
        price = power_prices[area].values.flatten()
        load = loads[area].values.flatten()
        hydro_revenue = np.sum(hydro * price)

        # Calculate weighted average price
        avg_price_market = np.average(price, weights=load)

        # Calculate value factor hydro area
        revenue_per_mwh = hydro_revenue / hydro.sum()
        value_factor = revenue_per_mwh / avg_price_market

        # Get number of weather years for averaging
        n_weather_years = hydro_generations[area].shape[1]

        # Store results (divide by n_weather_years for expected values)
        scenario_area_data[area] = {
            "hydro_generation_gwh": hydro.sum() / n_weather_years / 1000,  # MWh to GWh
            "hydro_revenue_meur": hydro_revenue / 1e6 / n_weather_years,  # Convert to M€
            "avg_market_price": avg_price_market,
            "revenue_per_mwh": revenue_per_mwh,
            "value_factor": value_factor,
        }

    # Calculate value factor uprated power plant
    for plant_id, plant_data in res_generations.items():
        hydro = plant_data["gen"].values.flatten()
        price = power_prices[plant_data["area"]].values.flatten()
        load = loads[plant_data["area"]].values.flatten()
        hydro_revenue = np.sum(hydro * price)

        avg_price_market = np.average(price, weights=load)
        revenue_per_mwh = hydro_revenue / hydro.sum()
        value_factor = revenue_per_mwh / avg_price_market

        # Get number of weather years for averaging
        n_weather_years = plant_data["gen"].shape[1]

        # Store results (divide by n_weather_years for expected values)
        scenario_uprate_data[plant_id] = {
            "hydro_generation_gwh": hydro.sum() / n_weather_years / 1000,  # MWh to GWh
            "hydro_revenue_meur": hydro_revenue / 1e6 / n_weather_years,  # Convert to M€
            "avg_market_price": avg_price_market,
            "revenue_per_mwh": revenue_per_mwh,
            "value_factor": value_factor,
        }

    area_hydro_results[scenario_name] = scenario_area_data
    uprate_hydro_results[scenario_name] = scenario_uprate_data

# ============================================================================
# Create visualizations
# ============================================================================

logger.info("\nCreating visualizations...")

# Plot 1: Revenue per MWh for Uprated Plants
logger.info("Creating revenue per MWh plot...")

# Prepare data for plotting - only uprated scenarios for bars
plant_names = []
scenarios_to_plot = []
revenue_per_mwh_data = []
value_factors_data = []

for scenario_name in SCENARIOS_UPRATED:
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    if scenario_name in uprate_hydro_results:
        scenario_data = uprate_hydro_results[scenario_name]
        for plant_name, data in scenario_data.items():
            plant_names.append(plant_name)
            scenarios_to_plot.append(short_name)
            revenue_per_mwh_data.append(data["revenue_per_mwh"])
            value_factors_data.append(data["value_factor"])

# Create DataFrame for easier plotting
df_uprate_rev = pd.DataFrame(
    {
        "Plant": plant_names,
        "Scenario": scenarios_to_plot,
        "Revenue_per_MWh": revenue_per_mwh_data,
        "Value_Factor": value_factors_data,
    }
)

# Create bar chart
fig, ax = plt.subplots(figsize=(20, 8))

# Get unique scenarios - only uprated ones for bars
unique_scenarios = [
    SCENARIO_LABELS[s] for s in SCENARIOS_UPRATED if SCENARIO_LABELS[s] in df_uprate_rev["Scenario"].unique()
]

# Group plants by area
plants_by_area = {}
for plant in df_uprate_rev["Plant"].unique():
    if plant in uprated_plant_info:
        area = uprated_plant_info[plant]["area"]
        if area not in plants_by_area:
            plants_by_area[area] = []
        plants_by_area[area].append(plant)

# Sort areas and plants within each area
sorted_areas = sorted(plants_by_area.keys())
plants_ordered = []
area_boundaries = [0]  # Track where each area starts for visual separation
for area in sorted_areas:
    area_plants = sorted(plants_by_area[area])
    plants_ordered.extend(area_plants)
    area_boundaries.append(len(plants_ordered))

# Set up bar positions
x = np.arange(len(plants_ordered))
width = 0.13  # Width of bars
n_scenarios = len(unique_scenarios)

# Color scheme for scenarios
colors = {
    # "B+": "#1f77b4",
    # "N-LLPS+": "#ff7f0e",
    # "OWN-LLPS+": "#2ca02c",
    # "OW-LLPS+": "#d62728",
    # "N-BA+": "#9467bd",
    # "OWN-BA+": "#8c564b",
    # "OW-BA+": "#e377c2",
    "B+": "#777777",
    "OW-BA+": "#3D7AA9",
    "OWN-BA+": "#408065",
    "N-BA+": "#E6974A",
    "OW-LLPS+": "#78AEDC",
    "OWN-LLPS+": "#66B08B",
    "N-LLPS+": "#F6C96C",

}


# Plot bars for each scenario
for i, scenario in enumerate(unique_scenarios):
    scenario_df = df_uprate_rev[df_uprate_rev["Scenario"] == scenario]

    # Create a list with revenue per MWh in the right order (matching plants_ordered)
    rev_values = []
    for plant in plants_ordered:
        plant_data = scenario_df[scenario_df["Plant"] == plant]
        if not plant_data.empty:
            rev_values.append(plant_data["Revenue_per_MWh"].values[0])
        else:
            rev_values.append(np.nan)  # Use NaN instead of 0 for missing data

    offset = (i - n_scenarios / 2 + 0.5) * width
    ax.bar(x + offset, rev_values, width, label=scenario, color=colors.get(scenario, "gray"))

# Add lines for non-uprated scenarios
no_uprate_legend_added = False
for scenario_name in SCENARIOS_NO_UPRATE:
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    if scenario_name in uprate_hydro_results:
        scenario_data = uprate_hydro_results[scenario_name]

        # Map non-uprated label to corresponding uprated bar scenario
        base_scenario = NO_UPRATE_TO_UPRATE_LABEL.get(short_name)

        # Find which bar index this scenario corresponds to
        if base_scenario and base_scenario in unique_scenarios:
            scenario_idx = unique_scenarios.index(base_scenario)
            offset = (scenario_idx - n_scenarios / 2 + 0.5) * width

            # Get x positions and values for this scenario's plants
            x_positions = []
            y_values = []
            for i, plant in enumerate(plants_ordered):
                if plant in scenario_data:
                    x_positions.append(i + offset)  # Add offset to align with bar
                    y_values.append(scenario_data[plant]["revenue_per_mwh"])

            if x_positions:
                # Use black color for non-uprated scenario lines
                for x_pos, y_val in zip(x_positions, y_values):
                    ax.hlines(
                        y_val,
                        x_pos - width / 2,
                        x_pos + width / 2,
                        colors="black",
                        linewidth=2.5,
                        zorder=10,
                        label="No Uprate" if not no_uprate_legend_added else "",
                    )
                    no_uprate_legend_added = True

# Add vertical lines to separate areas
for boundary in area_boundaries[1:-1]:
    ax.axvline(x=boundary - 0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

# Add area labels
for i, area in enumerate(sorted_areas):
    start = area_boundaries[i]
    end = area_boundaries[i + 1]
    mid = (start + end - 1) / 2
    ax.text(
        mid,
        ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        area,
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=11,
    )

# Customize plot
ax.set_xlabel("Uprated Plant (grouped by area)", fontsize=12, fontweight="bold")
ax.set_ylabel("Revenue per MWh (EUR/MWh)", fontsize=12, fontweight="bold")
ax.set_title("Revenue per MWh by Uprated Hydropower Plant and Scenario", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(plants_ordered, rotation=45, ha="right")
ax.legend(title="Scenario", loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Zoom in on y-axis - set limits based on data range
all_values = df_uprate_rev["Revenue_per_MWh"].values
min_rev = np.nanmin(all_values)
max_rev = np.nanmax(all_values)
margin = (max_rev - min_rev) * 0.1
ax.set_ylim(max(0, min_rev - margin), max_rev + margin)

plt.tight_layout()
output_file = paper_output_path / "uprated_plants_revenue_per_mwh.pdf"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
logger.info(f"Saved uprated plants revenue per MWh plot to: {output_file}")
plt.show()  # Display in Jupyter
plt.close()

# Across power plants, compute weighted average revenue for Nuclear, OWN and OW scenarios
weighted_avg_revenue_by_case = {}
scenario_cases = {"uprated": SCENARIOS_UPRATED, "non_uprated": SCENARIOS_NO_UPRATE}

for case_name, case_scenarios in scenario_cases.items():
    scenario_group_totals = {
        "Nuclear": {"weighted_revenue_sum": 0.0, "weight_sum": 0.0},
        "OWN": {"weighted_revenue_sum": 0.0, "weight_sum": 0.0},
        "OW": {"weighted_revenue_sum": 0.0, "weight_sum": 0.0},
    }

    for scen in case_scenarios:
        short_name = SCENARIO_LABELS.get(scen, scen)
        if short_name.startswith("N-"):
            group = "Nuclear"
        elif short_name.startswith("OWN-"):
            group = "OWN"
        elif short_name.startswith("OW-"):
            group = "OW"
        else:
            continue

        scen_data = uprate_hydro_results.get(scen, {})
        if not scen_data:
            continue

        prices = np.fromiter((d["revenue_per_mwh"] for d in scen_data.values()), dtype=float)
        weights = np.fromiter((d["hydro_generation_gwh"] for d in scen_data.values()), dtype=float)
        weight_sum = weights.sum()
        if weight_sum > 0:
            scenario_group_totals[group]["weighted_revenue_sum"] += float(np.dot(prices, weights))
            scenario_group_totals[group]["weight_sum"] += float(weight_sum)

    weighted_avg_revenue_by_case[case_name] = {
        group: (vals["weighted_revenue_sum"] / vals["weight_sum"] if vals["weight_sum"] > 0 else np.nan)
        for group, vals in scenario_group_totals.items()
    }

for case_name, case_results in weighted_avg_revenue_by_case.items():
    for group, avg_revenue in case_results.items():
        logger.info(f"Weighted average revenue across {case_name} plants ({group}): {avg_revenue:.2f} EUR/MWh")



# Plot 2: Total Revenue for Uprated Plants (not normalized by MWh)
logger.info("Creating total revenue plot...")

# Prepare data for total revenue plotting
plant_names_total = []
scenarios_to_plot_total = []
total_revenue_data = []

for scenario_name in SCENARIOS_UPRATED:
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    if scenario_name in uprate_hydro_results:
        scenario_data = uprate_hydro_results[scenario_name]
        for plant_name, data in scenario_data.items():
            plant_names_total.append(plant_name)
            scenarios_to_plot_total.append(short_name)
            total_revenue_data.append(data["hydro_revenue_meur"])

# Create DataFrame for total revenue
df_total_rev = pd.DataFrame(
    {
        "Plant": plant_names_total,
        "Scenario": scenarios_to_plot_total,
        "Total_Revenue_MEUR": total_revenue_data,
    }
)

# Create bar chart
fig, ax = plt.subplots(figsize=(18, 8))

# Plot bars for each scenario
for i, scenario in enumerate(unique_scenarios):
    scenario_df = df_total_rev[df_total_rev["Scenario"] == scenario]

    # Create a list with total revenue in the right order (matching plants_ordered)
    rev_values = []
    for plant in plants_ordered:
        plant_data = scenario_df[scenario_df["Plant"] == plant]
        if not plant_data.empty:
            rev_values.append(plant_data["Total_Revenue_MEUR"].values[0])
        else:
            rev_values.append(np.nan)

    offset = (i - n_scenarios / 2 + 0.5) * width
    ax.bar(x + offset, rev_values, width, label=scenario, color=colors.get(scenario, "gray"))

# Add lines for non-uprated scenarios
no_uprate_legend_added = False
for scenario_name in SCENARIOS_NO_UPRATE:
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    if scenario_name in uprate_hydro_results:
        scenario_data = uprate_hydro_results[scenario_name]

        # Map non-uprated label to corresponding uprated bar scenario
        base_scenario = NO_UPRATE_TO_UPRATE_LABEL.get(short_name)

        # Find which bar index this scenario corresponds to
        if base_scenario and base_scenario in unique_scenarios:
            scenario_idx = unique_scenarios.index(base_scenario)
            offset = (scenario_idx - n_scenarios / 2 + 0.5) * width

            # Get x positions and values
            x_positions = []
            y_values = []
            for i, plant in enumerate(plants_ordered):
                if plant in scenario_data:
                    x_positions.append(i + offset)
                    y_values.append(scenario_data[plant]["hydro_revenue_meur"])

            if x_positions:
                # Use black color for non-uprated scenario lines
                for x_pos, y_val in zip(x_positions, y_values):
                    ax.hlines(
                        y_val,
                        x_pos - width / 2,
                        x_pos + width / 2,
                        colors="black",
                        linewidth=2.5,
                        zorder=10,
                        label="No Uprate" if not no_uprate_legend_added else "",
                    )
                    no_uprate_legend_added = True

# Add vertical lines to separate areas
for boundary in area_boundaries[1:-1]:
    ax.axvline(x=boundary - 0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

# Add area labels
for i, area in enumerate(sorted_areas):
    start = area_boundaries[i]
    end = area_boundaries[i + 1]
    mid = (start + end - 1) / 2
    ax.text(
        mid,
        ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        area,
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=11,
    )

# Customize plot
ax.set_xlabel("Uprated Plant (grouped by area)", fontsize=12, fontweight="bold")
ax.set_ylabel("Total Annual Revenue (M€)", fontsize=12, fontweight="bold")
ax.set_title("Total Annual Revenue by Uprated Hydropower Plant and Scenario", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(plants_ordered, rotation=45, ha="right")
ax.legend(title="Scenario", loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Zoom in on y-axis - set limits based on data range
all_values_total = df_total_rev["Total_Revenue_MEUR"].values
min_rev_total = np.nanmin(all_values_total)
max_rev_total = np.nanmax(all_values_total)
margin_total = (max_rev_total - min_rev_total) * 0.1
ax.set_ylim(max(0, min_rev_total - margin_total), max_rev_total + margin_total)

plt.tight_layout()
output_file = paper_output_path / "uprated_plants_total_revenue.pdf"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
logger.info(f"Saved uprated plants total revenue plot to: {output_file}")
plt.show()  # Display in Jupyter
plt.close()

# Plot 3: Comparison of Area-level vs Plant-level Value Factors
logger.info("Creating comparison plot of area vs plant-level value factors...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

# Left panel: Area-level value factors (all hydro) - BAR CHART
x_pos = np.arange(len(NO_AREAS))
width = 0.12

for i, scenario in enumerate(unique_scenarios):
    vf_values = []
    for area in NO_AREAS:
        for scenario_full, scenario_data in area_hydro_results.items():
            if SCENARIO_LABELS.get(scenario_full) == scenario and area in scenario_data:
                vf_values.append(scenario_data[area]["value_factor"])
                break
        else:
            vf_values.append(np.nan)

    offset = (i - n_scenarios / 2 + 0.5) * width
    ax1.bar(x_pos + offset, vf_values, width, label=scenario, color=colors.get(scenario, "gray"))

# Add horizontal lines for non-uprated scenarios on left panel
for scenario_name in SCENARIOS_NO_UPRATE:
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    if scenario_name in area_hydro_results:
        scenario_data = area_hydro_results[scenario_name]

        # Map non-uprated label to corresponding uprated bar scenario
        base_scenario = NO_UPRATE_TO_UPRATE_LABEL.get(short_name)

        # Find which bar index this scenario corresponds to
        if base_scenario and base_scenario in unique_scenarios:
            scenario_idx = unique_scenarios.index(base_scenario)
            offset = (scenario_idx - n_scenarios / 2 + 0.5) * width

            # Plot lines for each area
            for i, area in enumerate(NO_AREAS):
                if area in scenario_data:
                    vf_value = scenario_data[area]["value_factor"]
                    ax1.hlines(
                        vf_value,
                        i + offset - width / 2,
                        i + offset + width / 2,
                        colors="black",
                        linewidth=2.0,
                        zorder=10,
                    )

ax1.set_xlabel("Area", fontsize=12, fontweight="bold")
ax1.set_ylabel("Value Factor", fontsize=12, fontweight="bold")
ax1.set_title("All Hydropower Value Factor by Area", fontsize=13, fontweight="bold")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(NO_AREAS)
ax1.legend(title="Scenario", fontsize=9, loc="upper right")
ax1.grid(axis="y", alpha=0.3)
ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)

# Zoom in on y-axis for left panel
area_vf_values = []
for scenario_full, scenario_data in area_hydro_results.items():
    for area, data in scenario_data.items():
        if area in NO_AREAS:
            area_vf_values.append(data["value_factor"])
if area_vf_values:
    min_vf = np.nanmin(area_vf_values)
    max_vf = np.nanmax(area_vf_values)
    margin = (max_vf - min_vf) * 0.1
    ax1.set_ylim(max(0, min_vf - margin), max_vf + margin)

# Right panel: Plant-level value factors for uprated plants
x_plants = np.arange(len(plants_ordered))
plant_width = 0.12

for i, scenario in enumerate(unique_scenarios):
    scenario_df = df_uprate_rev[df_uprate_rev["Scenario"] == scenario]

    vf_values = []
    for plant in plants_ordered:
        plant_data = scenario_df[scenario_df["Plant"] == plant]
        if not plant_data.empty:
            vf_values.append(plant_data["Value_Factor"].values[0])
        else:
            vf_values.append(np.nan)

    offset = (i - n_scenarios / 2 + 0.5) * plant_width
    ax2.bar(x_plants + offset, vf_values, plant_width, label=scenario, color=colors.get(scenario, "gray"))

# Add horizontal lines for non-uprated scenarios on right panel
for scenario_name in SCENARIOS_NO_UPRATE:
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    if scenario_name in uprate_hydro_results:
        scenario_data = uprate_hydro_results[scenario_name]
        base_scenario = NO_UPRATE_TO_UPRATE_LABEL.get(short_name)

        if base_scenario and base_scenario in unique_scenarios:
            scenario_idx = unique_scenarios.index(base_scenario)
            offset = (scenario_idx - n_scenarios / 2 + 0.5) * plant_width

            for i, plant in enumerate(plants_ordered):
                if plant in scenario_data:
                    vf_value = scenario_data[plant]["value_factor"]
                    ax2.hlines(
                        vf_value,
                        i + offset - plant_width / 2,
                        i + offset + plant_width / 2,
                        colors="black",
                        linewidth=2.0,
                        zorder=10,
                    )

# Add vertical lines to separate areas on right panel
for boundary in area_boundaries[1:-1]:
    ax2.axvline(x=boundary - 0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

ax2.set_xlabel("Uprated Plant (grouped by area)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Value Factor", fontsize=12, fontweight="bold")
ax2.set_title("Uprated Plant Value Factor by Scenario", fontsize=13, fontweight="bold")
ax2.set_xticks(x_plants)
ax2.set_xticklabels(plants_ordered, rotation=45, ha="right")
ax2.legend(title="Scenario", fontsize=9, loc="upper right")
ax2.grid(axis="y", alpha=0.3)
ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)

# Zoom in on y-axis for right panel
plant_vf_values = df_uprate_rev["Value_Factor"].values
if len(plant_vf_values) > 0:
    min_vf_p = np.nanmin(plant_vf_values)
    max_vf_p = np.nanmax(plant_vf_values)
    margin_p = (max_vf_p - min_vf_p) * 0.1
    ax2.set_ylim(max(0, min_vf_p - margin_p), max_vf_p + margin_p)

plt.tight_layout()
output_file = paper_output_path / "hydro_value_factor_area_vs_plant.pdf"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
logger.info(f"Saved area vs plant value factor plot to: {output_file}")
plt.show()  # Display in Jupyter
plt.close()
