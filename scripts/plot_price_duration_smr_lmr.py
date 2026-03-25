"""
Plot price duration curves for NO2 and NO4 across SMR and LMR scenarios.

Creates duration curve visualizations showing how electricity prices are distributed
across different nuclear reactor configurations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Baseline scenario
BASELINE_SCENARIO = "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF"

# SMR and LMR scenarios
SCENARIOS = [
    BASELINE_SCENARIO,
    # SMR scenarios - BA
    "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    # LMR scenarios - BA
    "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
    # SMR scenarios - LLPS
    "SMR300LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR900LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR1200LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    # LMR scenarios - LLPS
    "LMR2000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR3000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR4000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
]

# Scenario labels for display
SCENARIO_LABELS = {
    "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF": "BASELINE",
    "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF": "SMR300_BA",
    "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF": "SMR600_BA",
    "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF": "SMR900_BA",
    "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF": "SMR1200_BA",
    "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF": "SMR1600_BA",
    "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF": "LMR2000_BA",
    "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF": "LMR3000_BA",
    "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF": "LMR4000_BA",
    "SMR300LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF": "SMR300_LLPS",
    "SMR600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF": "SMR600_LLPS",
    "SMR900LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF": "SMR900_LLPS",
    "SMR1200LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF": "SMR1200_LLPS",
    "SMR1600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF": "SMR1600_LLPS",
    "LMR2000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF": "LMR2000_LLPS",
    "LMR3000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF": "LMR3000_LLPS",
    "LMR4000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF": "LMR4000_LLPS",
}

# Color schemes
REACTOR_COLORS = {
    "SMR": "#1f77b4",  # Blue
    "LMR": "#d62728",  # Red
    "BASELINE": "#2ca02c",  # Green
}

# Areas to analyze
AREAS = ["NO2", "NO4"]

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)

# Load scenarios
logger.info("Loading scenarios...")
scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

logger.info(f"Loaded {len(scenarios)} scenarios")

# Import after loading scenarios
from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

# ============================================================================
# Extract price data for each scenario and area
# ============================================================================

logger.info("\nExtracting price data...")

price_data = {}

for scenario_name, scenario in scenarios.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    logger.info(f"  Processing {short_name}...")

    try:
        busbars_dict = scenario.get_busbars()
        price_data[short_name] = {}

        for area in AREAS:
            if area in busbars_dict:
                busbar = busbars_dict[area]
                df_price = df_from_pyltm_result(busbar.market_result_price())

                # Flatten across all weather years and time periods
                prices = df_price.values.flatten()

                # Sort in descending order for duration curve
                prices_sorted = np.sort(prices)[::-1]

                price_data[short_name][area] = prices_sorted
                logger.info(f"    {area}: {len(prices_sorted)} price points, mean={prices.mean():.2f} €/MWh")

    except Exception as e:
        logger.error(f"  Failed to process {short_name}: {e}")
        continue

# ============================================================================
# Create duration curve plots
# ============================================================================

logger.info("\nCreating price duration curve plots...")

# Separate scenarios by reactor type and load profile
smr_ba_scenarios = [s for s in SCENARIO_LABELS.values() if "SMR" in s and "_BA" in s]
lmr_ba_scenarios = [s for s in SCENARIO_LABELS.values() if "LMR" in s and "_BA" in s]
smr_llps_scenarios = [s for s in SCENARIO_LABELS.values() if "SMR" in s and "_LLPS" in s]
lmr_llps_scenarios = [s for s in SCENARIO_LABELS.values() if "LMR" in s and "_LLPS" in s]

# Sort by capacity
smr_ba_scenarios.sort(key=lambda x: int(x.split("_")[0][3:]))
lmr_ba_scenarios.sort(key=lambda x: int(x.split("_")[0][3:]))
smr_llps_scenarios.sort(key=lambda x: int(x.split("_")[0][3:]))
lmr_llps_scenarios.sort(key=lambda x: int(x.split("_")[0][3:]))


# Function to create duration curve plot
def plot_duration_curves(ax, area, scenarios, color, label_prefix, include_baseline=True):
    """Plot duration curves for a set of scenarios."""

    if include_baseline and "BASELINE" in price_data and area in price_data["BASELINE"]:
        prices = price_data["BASELINE"][area]
        n_points = len(prices)
        percentiles = np.arange(n_points) / n_points * 100
        ax.plot(percentiles, prices, color=REACTOR_COLORS["BASELINE"], linewidth=2.5, label="BASELINE", alpha=0.9)

    for i, scenario in enumerate(scenarios):
        if scenario in price_data and area in price_data[scenario]:
            prices = price_data[scenario][area]
            n_points = len(prices)
            percentiles = np.arange(n_points) / n_points * 100

            # Extract capacity for label
            capacity = scenario.split("_")[0][3:]

            # Vary alpha and linewidth based on capacity
            alpha = 0.5 + (i / len(scenarios)) * 0.4
            linewidth = 1.5 + (i / len(scenarios)) * 1.0

            ax.plot(
                percentiles,
                prices,
                color=color,
                linewidth=linewidth,
                label=f"{label_prefix}{capacity}MW",
                alpha=alpha,
                linestyle="-",
            )


# Figure 1: BA scenarios - NO2 and NO4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# NO2 - BA
plot_duration_curves(ax1, "NO2", smr_ba_scenarios, REACTOR_COLORS["SMR"], "SMR ")
plot_duration_curves(ax1, "NO2", lmr_ba_scenarios, REACTOR_COLORS["LMR"], "LMR ", include_baseline=False)

ax1.set_xlabel("Duration (%)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Price (€/MWh)", fontsize=12, fontweight="bold")
ax1.set_title("NO2 Price Duration Curve - BA Scenarios", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9, loc="best", ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 100])
ax1.set_ylim([-10, 150])

# NO4 - BA
plot_duration_curves(ax2, "NO4", smr_ba_scenarios, REACTOR_COLORS["SMR"], "SMR ")
plot_duration_curves(ax2, "NO4", lmr_ba_scenarios, REACTOR_COLORS["LMR"], "LMR ", include_baseline=False)

ax2.set_xlabel("Duration (%)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Price (€/MWh)", fontsize=12, fontweight="bold")
ax2.set_title("NO4 Price Duration Curve - BA Scenarios", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, loc="best", ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 100])
ax2.set_ylim([-10, 150])

plt.suptitle(
    "Price Duration Curves - Baseload Addition (BA)\nBlue: SMR (Distributed) | Red: LMR (Concentrated)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_file_ba = paper_output_path / "price_duration_curves_ba.pdf"
plt.savefig(output_file_ba, dpi=300, bbox_inches="tight")
logger.info(f"Saved BA duration curves to: {output_file_ba}")

# Figure 2: LLPS scenarios - NO2 and NO4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# NO2 - LLPS
plot_duration_curves(ax1, "NO2", smr_llps_scenarios, REACTOR_COLORS["SMR"], "SMR ")
plot_duration_curves(ax1, "NO2", lmr_llps_scenarios, REACTOR_COLORS["LMR"], "LMR ", include_baseline=False)

ax1.set_xlabel("Duration (%)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Price (€/MWh)", fontsize=12, fontweight="bold")
ax1.set_title("NO2 Price Duration Curve - LLPS Scenarios", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9, loc="best", ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 100])
ax1.set_ylim([-10, 150])

# NO4 - LLPS
plot_duration_curves(ax2, "NO4", smr_llps_scenarios, REACTOR_COLORS["SMR"], "SMR ")
plot_duration_curves(ax2, "NO4", lmr_llps_scenarios, REACTOR_COLORS["LMR"], "LMR ", include_baseline=False)

ax2.set_xlabel("Duration (%)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Price (€/MWh)", fontsize=12, fontweight="bold")
ax2.set_title("NO4 Price Duration Curve - LLPS Scenarios", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, loc="best", ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 100])
ax2.set_ylim([-10, 150])

plt.suptitle(
    "Price Duration Curves - Linear Load Profile Scaling (LLPS)\nBlue: SMR (Distributed) | Red: LMR (Concentrated)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_file_llps = paper_output_path / "price_duration_curves_llps.pdf"
plt.savefig(output_file_llps, dpi=300, bbox_inches="tight")
logger.info(f"Saved LLPS duration curves to: {output_file_llps}")

# Figure 3: Combined comparison - SMR vs LMR at maximum capacity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# NO2 comparison
if "BASELINE" in price_data and "NO2" in price_data["BASELINE"]:
    prices = price_data["BASELINE"]["NO2"]
    n_points = len(prices)
    percentiles = np.arange(n_points) / n_points * 100
    ax1.plot(percentiles, prices, color=REACTOR_COLORS["BASELINE"], linewidth=3, label="BASELINE", alpha=0.9)

# Plot maximum capacity scenarios
max_scenarios = [
    ("SMR1600_BA", "SMR 8000MW BA", REACTOR_COLORS["SMR"], "-"),
    ("LMR4000_BA", "LMR 8000MW BA", REACTOR_COLORS["LMR"], "-"),
    ("SMR1600_LLPS", "SMR 8000MW LLPS", REACTOR_COLORS["SMR"], "--"),
    ("LMR4000_LLPS", "LMR 8000MW LLPS", REACTOR_COLORS["LMR"], "--"),
]

for scenario, label, color, linestyle in max_scenarios:
    if scenario in price_data and "NO2" in price_data[scenario]:
        prices = price_data[scenario]["NO2"]
        n_points = len(prices)
        percentiles = np.arange(n_points) / n_points * 100
        ax1.plot(percentiles, prices, color=color, linewidth=2.5, label=label, alpha=0.8, linestyle=linestyle)

ax1.set_xlabel("Duration (%)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Price (€/MWh)", fontsize=12, fontweight="bold")
ax1.set_title("NO2 Price Duration - Maximum Capacity Comparison", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10, loc="best")
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 100])
ax1.set_ylim([-10, 150])

# NO4 comparison
if "BASELINE" in price_data and "NO4" in price_data["BASELINE"]:
    prices = price_data["BASELINE"]["NO4"]
    n_points = len(prices)
    percentiles = np.arange(n_points) / n_points * 100
    ax2.plot(percentiles, prices, color=REACTOR_COLORS["BASELINE"], linewidth=3, label="BASELINE", alpha=0.9)

for scenario, label, color, linestyle in max_scenarios:
    if scenario in price_data and "NO4" in price_data[scenario]:
        prices = price_data[scenario]["NO4"]
        n_points = len(prices)
        percentiles = np.arange(n_points) / n_points * 100
        ax2.plot(percentiles, prices, color=color, linewidth=2.5, label=label, alpha=0.8, linestyle=linestyle)

ax2.set_xlabel("Duration (%)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Price (€/MWh)", fontsize=12, fontweight="bold")
ax2.set_title("NO4 Price Duration - Maximum Capacity Comparison", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, loc="best")
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 100])
ax2.set_ylim([-10, 150])

plt.suptitle(
    "Price Duration Curves - SMR vs LMR at 8000 MW\nSolid: BA | Dashed: LLPS", fontsize=14, fontweight="bold", y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_file_comp = paper_output_path / "price_duration_curves_comparison.pdf"
plt.savefig(output_file_comp, dpi=300, bbox_inches="tight")
logger.info(f"Saved comparison duration curves to: {output_file_comp}")

# ============================================================================
# Create summary statistics
# ============================================================================

logger.info("\nCalculating summary statistics...")

summary_data = []

for scenario_name in ["BASELINE"] + smr_ba_scenarios + lmr_ba_scenarios + smr_llps_scenarios + lmr_llps_scenarios:
    if scenario_name in price_data:
        for area in AREAS:
            if area in price_data[scenario_name]:
                prices = price_data[scenario_name][area]

                summary_data.append(
                    {
                        "Scenario": scenario_name,
                        "Area": area,
                        "Mean_Price": prices.mean(),
                        "Median_Price": np.median(prices),
                        "P95_Price": np.percentile(prices, 95),
                        "P5_Price": np.percentile(prices, 5),
                        "Max_Price": prices.max(),
                        "Min_Price": prices.min(),
                        "Std_Price": prices.std(),
                    }
                )

df_summary = pd.DataFrame(summary_data)
output_csv = paper_output_path / "price_duration_statistics.csv"
df_summary.to_csv(output_csv, index=False, float_format="%.2f")
logger.info(f"Saved summary statistics to: {output_csv}")

logger.info("\n" + "=" * 80)
logger.info("✓ Script completed successfully!")
logger.info("=" * 80)
logger.info("Generated files:")
logger.info(f"  - {output_file_ba.name}")
logger.info(f"  - {output_file_llps.name}")
logger.info(f"  - {output_file_comp.name}")
logger.info(f"  - {output_csv.name}")
logger.info("=" * 80)
