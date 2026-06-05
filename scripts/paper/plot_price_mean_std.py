"""
Plot mean and std dev of electricity prices for OW, N, OWN scenarios.
Shows LLPS and BA scenarios together in horizontal subplots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

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
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF": "BASELINE",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "LLPS_N",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "LLPS_OWN",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "LLPS_OW",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "BA_N",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "BA_OWN",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "BA_OW",
}

# Group scenarios by type
SCENARIO_GROUPS = {
    "N": ["LLPS_N", "BA_N"],
    "OWN": ["LLPS_OWN", "BA_OWN"],
    "OW": ["LLPS_OW", "BA_OW"],
}

# Norwegian areas to aggregate
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# All Nordic areas for area-by-area analysis
ALL_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4", "DK1", "DK2", "FI"]

# Colors
COLORS = {
    "LLPS": "#1f77b4",  # Blue
    "BA": "#ff7f0e",  # Orange
    "BASELINE": "#2ca02c",  # Green
}

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)

# Load scenarios
scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

logger.info(f"Loaded {len(scenarios)} scenarios")

# Collect price statistics for each scenario
price_stats = {}
price_stats_by_area = {}

for scenario_name, scenario in scenarios.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    logger.info(f"Processing scenario: {short_name}")

    try:
        busbar_names = set(scenario.get_busbar_names())

        # Aggregate prices across Norwegian areas with volume weighting
        all_prices = []
        all_volumes = []

        # Store area-specific statistics
        area_stats = {}

        for area in NO_AREAS:
            if area in busbar_names:
                df_price = scenario.get_prices_for_busbar(area)
                df_load = scenario.get_load_for_busbar(area)

                prices = df_price.values.flatten()
                volumes = df_load.values.flatten()

                all_prices.append(prices)
                all_volumes.append(volumes)

        # Calculate statistics for all areas
        for area in ALL_AREAS:
            if area in busbar_names:
                df_price = scenario.get_prices_for_busbar(area)
                df_load = scenario.get_load_for_busbar(area)

                prices = df_price.values.flatten()
                volumes = df_load.values.flatten()

                # Volume-weighted mean
                volume_weighted_mean = np.average(prices, weights=volumes)
                # Standard deviation (not weighted)
                std_price = np.std(prices)

                area_stats[area] = {
                    "mean": volume_weighted_mean,
                    "std": std_price,
                }

        if all_prices and all_volumes:
            # Concatenate all prices and volumes
            combined_prices = np.concatenate(all_prices)
            combined_volumes = np.concatenate(all_volumes)

            # Calculate volume-weighted statistics for Norwegian areas
            volume_weighted_mean = np.average(combined_prices, weights=combined_volumes)
            std_price = np.std(combined_prices)

            price_stats[short_name] = {
                "mean": volume_weighted_mean,
                "std": std_price,
            }

            price_stats_by_area[short_name] = area_stats

            logger.info(f"  {short_name}: Mean={volume_weighted_mean:.2f} €/MWh, Std={std_price:.2f} €/MWh")
        else:
            logger.warning(f"  {short_name}: No price data available")

    except Exception as e:
        logger.error(f"Failed to process {scenario_name}: {e}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Define x positions for grouped bars
x_groups = np.arange(len(SCENARIO_GROUPS))
bar_width = 0.35
x_llps = x_groups - bar_width / 2
x_ba = x_groups + bar_width / 2

# Plot 1: Mean prices
ax = axes[0]
mean_llps = []
mean_ba = []

for group_name in ["N", "OWN", "OW"]:
    scenarios_in_group = SCENARIO_GROUPS[group_name]
    llps_scenario = scenarios_in_group[0]  # LLPS version
    ba_scenario = scenarios_in_group[1]  # BA version

    mean_llps.append(price_stats.get(llps_scenario, {}).get("mean", 0))
    mean_ba.append(price_stats.get(ba_scenario, {}).get("mean", 0))

bars1 = ax.bar(x_llps, mean_llps, bar_width, label="LLPS", color=COLORS["LLPS"], alpha=0.8)
bars2 = ax.bar(x_ba, mean_ba, bar_width, label="BA", color=COLORS["BA"], alpha=0.8)

# Add baseline reference line if available
if "BASELINE" in price_stats:
    baseline_mean = price_stats["BASELINE"]["mean"]
    ax.axhline(
        y=baseline_mean,
        color=COLORS["BASELINE"],
        linestyle="--",
        linewidth=2,
        label=f"BASELINE ({baseline_mean:.1f} €/MWh)",
        alpha=0.7,
    )

ax.set_xlabel("Scenario Type", fontsize=12)
ax.set_ylabel("Mean Price (€/MWh)", fontsize=12)
ax.set_title("Mean Electricity Price by Scenario", fontsize=14, fontweight="bold")
ax.set_xticks(x_groups)
ax.set_xticklabels(["N (Nuclear)", "OWN (Offshore + Nuclear)", "OW (Offshore)"])
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=9)

# Plot 2: Standard deviation
ax = axes[1]
std_llps = []
std_ba = []

for group_name in ["N", "OWN", "OW"]:
    scenarios_in_group = SCENARIO_GROUPS[group_name]
    llps_scenario = scenarios_in_group[0]
    ba_scenario = scenarios_in_group[1]

    std_llps.append(price_stats.get(llps_scenario, {}).get("std", 0))
    std_ba.append(price_stats.get(ba_scenario, {}).get("std", 0))

bars1 = ax.bar(x_llps, std_llps, bar_width, label="LLPS", color=COLORS["LLPS"], alpha=0.8)
bars2 = ax.bar(x_ba, std_ba, bar_width, label="BA", color=COLORS["BA"], alpha=0.8)

# Add baseline reference line if available
if "BASELINE" in price_stats:
    baseline_std = price_stats["BASELINE"]["std"]
    ax.axhline(
        y=baseline_std,
        color=COLORS["BASELINE"],
        linestyle="--",
        linewidth=2,
        label=f"BASELINE ({baseline_std:.1f} €/MWh)",
        alpha=0.7,
    )

ax.set_xlabel("Scenario Type", fontsize=12)
ax.set_ylabel("Price Std Dev (€/MWh)", fontsize=12)
ax.set_title("Price Standard Deviation by Scenario", fontsize=14, fontweight="bold")
ax.set_xticks(x_groups)
ax.set_xticklabels(["N (Nuclear)", "OWN (Offshore + Nuclear)", "OW (Offshore)"])
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()

# Save figure
output_file = paper_output_path / "price_mean_std_comparison.pdf"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
logger.info(f"\nSaved plot to: {output_file}")

# ============================================================================
# FIGURE 2: Area-by-area comparison (including TOTAL system average)
# ============================================================================

# Extended area list with TOTAL at the end
AREAS_WITH_TOTAL = ALL_AREAS + ["TOTAL"]

fig2, axes2 = plt.subplots(1, 2, figsize=(20, 6))

# Plot 1: Mean prices by area
ax = axes2[0]

# Prepare data for each scenario type
bar_width = 0.12
x_pos = np.arange(len(AREAS_WITH_TOTAL))

# Define offsets for each scenario
offsets = {
    "LLPS_N": -2.5 * bar_width,
    "LLPS_OWN": -1.5 * bar_width,
    "LLPS_OW": -0.5 * bar_width,
    "BA_N": 0.5 * bar_width,
    "BA_OWN": 1.5 * bar_width,
    "BA_OW": 2.5 * bar_width,
}

colors_scenarios = {
    "LLPS_N": "#1f77b4",
    "LLPS_OWN": "#2ca02c",
    "LLPS_OW": "#17becf",
    "BA_N": "#ff7f0e",
    "BA_OWN": "#d62728",
    "BA_OW": "#ff9896",
}

for scenario_name, offset in offsets.items():
    if scenario_name in price_stats_by_area:
        mean_values = []
        for area in ALL_AREAS:
            area_data = price_stats_by_area[scenario_name].get(area, {})
            mean_values.append(area_data.get("mean", 0))

        # Add TOTAL system average from price_stats
        if scenario_name in price_stats:
            mean_values.append(price_stats[scenario_name].get("mean", 0))
        else:
            mean_values.append(0)

        ax.bar(
            x_pos + offset,
            mean_values,
            bar_width,
            label=scenario_name.replace("_", " "),
            color=colors_scenarios.get(scenario_name, "#333333"),
            alpha=0.8,
        )

# Add baseline if available
if "BASELINE" in price_stats_by_area:
    baseline_means = []
    for area in ALL_AREAS:
        area_data = price_stats_by_area["BASELINE"].get(area, {})
        baseline_means.append(area_data.get("mean", 0))

    # Add TOTAL system average from price_stats
    if "BASELINE" in price_stats:
        baseline_means.append(price_stats["BASELINE"].get("mean", 0))
    else:
        baseline_means.append(0)

    ax.plot(
        x_pos,
        baseline_means,
        color=COLORS["BASELINE"],
        linestyle="--",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="BASELINE",
        alpha=0.8,
    )

ax.set_xlabel("Area", fontsize=12)
ax.set_ylabel("Volume-Weighted Mean Price (€/MWh)", fontsize=12)
ax.set_title("Mean Electricity Price by Area and Scenario", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(AREAS_WITH_TOTAL, rotation=0)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis="y")

# Plot 2: Std dev by area
ax = axes2[1]

for scenario_name, offset in offsets.items():
    if scenario_name in price_stats_by_area:
        std_values = []
        for area in ALL_AREAS:
            area_data = price_stats_by_area[scenario_name].get(area, {})
            std_values.append(area_data.get("std", 0))

        # Add TOTAL system std dev from price_stats
        if scenario_name in price_stats:
            std_values.append(price_stats[scenario_name].get("std", 0))
        else:
            std_values.append(0)

        ax.bar(
            x_pos + offset,
            std_values,
            bar_width,
            label=scenario_name.replace("_", " "),
            color=colors_scenarios.get(scenario_name, "#333333"),
            alpha=0.8,
        )

# Add baseline if available
if "BASELINE" in price_stats_by_area:
    baseline_stds = []
    for area in ALL_AREAS:
        area_data = price_stats_by_area["BASELINE"].get(area, {})
        baseline_stds.append(area_data.get("std", 0))

    # Add TOTAL system std dev from price_stats
    if "BASELINE" in price_stats:
        baseline_stds.append(price_stats["BASELINE"].get("std", 0))
    else:
        baseline_stds.append(0)

    ax.plot(
        x_pos,
        baseline_stds,
        color=COLORS["BASELINE"],
        linestyle="--",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="BASELINE",
        alpha=0.8,
    )

ax.set_xlabel("Area", fontsize=12)
ax.set_ylabel("Price Std Dev (€/MWh)", fontsize=12)
ax.set_title("Price Standard Deviation by Area and Scenario", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(AREAS_WITH_TOTAL, rotation=0)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

# Save second figure
output_file2 = paper_output_path / "price_mean_std_by_area.pdf"
plt.savefig(output_file2, dpi=300, bbox_inches="tight")
logger.info(f"Saved area-by-area plot to: {output_file2}")

# ============================================================================
# FIGURE 3: Norwegian areas only comparison
# ============================================================================

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Mean prices for Norwegian areas
ax = axes3[0]

# Prepare data for each scenario type - Norwegian areas only
bar_width_no = 0.08
x_pos_no = np.arange(len(NO_AREAS))

for scenario_name, offset in offsets.items():
    if scenario_name in price_stats_by_area:
        mean_values_no = []
        for area in NO_AREAS:
            area_data = price_stats_by_area[scenario_name].get(area, {})
            mean_values_no.append(area_data.get("mean", 0))

        ax.bar(
            x_pos_no + offset,
            mean_values_no,
            bar_width_no,
            label=scenario_name.replace("_", " "),
            color=colors_scenarios.get(scenario_name, "#333333"),
            alpha=0.8,
        )

# Add baseline if available
if "BASELINE" in price_stats_by_area:
    baseline_means_no = []
    for area in NO_AREAS:
        area_data = price_stats_by_area["BASELINE"].get(area, {})
        baseline_means_no.append(area_data.get("mean", 0))

    ax.plot(
        x_pos_no,
        baseline_means_no,
        color=COLORS["BASELINE"],
        linestyle="--",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="BASELINE",
        alpha=0.8,
    )

ax.set_xlabel("Norwegian Area", fontsize=12)
ax.set_ylabel("Volume-Weighted Mean Price (€/MWh)", fontsize=12)
ax.set_title("Mean Electricity Price - Norwegian Areas Only", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos_no)
ax.set_xticklabels(NO_AREAS, rotation=0)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis="y")

# Plot 2: Std dev for Norwegian areas
ax = axes3[1]

for scenario_name, offset in offsets.items():
    if scenario_name in price_stats_by_area:
        std_values_no = []
        for area in NO_AREAS:
            area_data = price_stats_by_area[scenario_name].get(area, {})
            std_values_no.append(area_data.get("std", 0))

        ax.bar(
            x_pos_no + offset,
            std_values_no,
            bar_width_no,
            label=scenario_name.replace("_", " "),
            color=colors_scenarios.get(scenario_name, "#333333"),
            alpha=0.8,
        )

# Add baseline if available
if "BASELINE" in price_stats_by_area:
    baseline_stds_no = []
    for area in NO_AREAS:
        area_data = price_stats_by_area["BASELINE"].get(area, {})
        baseline_stds_no.append(area_data.get("std", 0))

    ax.plot(
        x_pos_no,
        baseline_stds_no,
        color=COLORS["BASELINE"],
        linestyle="--",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="BASELINE",
        alpha=0.8,
    )

ax.set_xlabel("Norwegian Area", fontsize=12)
ax.set_ylabel("Price Std Dev (€/MWh)", fontsize=12)
ax.set_title("Price Standard Deviation - Norwegian Areas Only", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos_no)
ax.set_xticklabels(NO_AREAS, rotation=0)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

# Save third figure
output_file3 = paper_output_path / "price_mean_std_norwegian_areas.pdf"
plt.savefig(output_file3, dpi=300, bbox_inches="tight")
logger.info(f"Saved Norwegian areas plot to: {output_file3}")

# Create summary table
summary_data = []
for group_name in ["N", "OWN", "OW"]:
    for scenario in SCENARIO_GROUPS[group_name]:
        if scenario in price_stats:
            stats = price_stats[scenario]
            load_type = "LLPS" if "LLPS" in scenario else "BA"
            summary_data.append(
                {
                    "Group": group_name,
                    "Load": load_type,
                    "Scenario": scenario,
                    "Mean (€/MWh, vol-weighted)": stats["mean"],
                    "Std Dev (€/MWh)": stats["std"],
                }
            )

if "BASELINE" in price_stats:
    summary_data.append(
        {
            "Group": "BASELINE",
            "Load": "-",
            "Scenario": "BASELINE",
            "Mean (€/MWh, vol-weighted)": price_stats["BASELINE"]["mean"],
            "Std Dev (€/MWh)": price_stats["BASELINE"]["std"],
        }
    )

df_summary = pd.DataFrame(summary_data)

# Save summary table
output_csv = paper_output_path / "price_mean_std_summary.csv"
df_summary.to_csv(output_csv, index=False)
logger.info(f"Saved summary table to: {output_csv}")

# Create area-by-area summary table
area_summary_data = []
for scenario_name in ["BASELINE", "LLPS_N", "LLPS_OWN", "LLPS_OW", "BA_N", "BA_OWN", "BA_OW"]:
    if scenario_name in price_stats_by_area:
        for area in ALL_AREAS:
            if area in price_stats_by_area[scenario_name]:
                stats = price_stats_by_area[scenario_name][area]
                area_summary_data.append(
                    {
                        "Scenario": scenario_name,
                        "Area": area,
                        "Mean (€/MWh, vol-weighted)": stats["mean"],
                        "Std Dev (€/MWh)": stats["std"],
                    }
                )

df_area_summary = pd.DataFrame(area_summary_data)

# Save area summary table
output_area_csv = paper_output_path / "price_mean_std_by_area_summary.csv"
df_area_summary.to_csv(output_area_csv, index=False)
logger.info(f"Saved area summary table to: {output_area_csv}")

# Print summary
print("\n" + "=" * 80)
print("PRICE STATISTICS SUMMARY")
print("=" * 80)
print(df_summary.to_string(index=False))
print("=" * 80)

plt.show()
