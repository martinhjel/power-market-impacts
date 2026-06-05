"""
Plot price duration curves for NO2 and NO4 across SMR and LMR scenarios.

Creates duration curve visualizations showing how electricity prices are distributed
across different nuclear reactor configurations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF": "BASELINE30",
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
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
CAPACITY_LINE_AREA_ORDER = ["NO2", "NO1", "NO5", "NO3", "NO4", "NO"]
CAPACITY_LINE_AREA_COLORS = {
    "NO2": "#ff7f0e",
    "NO1": "#1f77b4",
    "NO5": "#9467bd",
    "NO3": "#2ca02c",
    "NO4": "#d62728",
    "NO": "#000000",
}

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

# ============================================================================
# Extract price data for each scenario and area
# ============================================================================

logger.info("\nExtracting price data...")

price_data = {}

for scenario_name, scenario in scenarios.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    logger.info(f"  Processing {short_name}...")

    try:
        busbar_names = set(scenario.get_busbar_names())
        price_data[short_name] = {}

        for area in AREAS:
            if area in busbar_names:
                df_price = scenario.get_prices_for_busbar(area)

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

# ============================================================================
# Create mean price vs installed nuclear capacity line plot
# ============================================================================

logger.info("\nCreating BA mean price vs installed nuclear capacity plot...")


def parse_reactor_capacity(short_name: str) -> tuple[str, int] | None:
    """Return reactor type and total Norwegian nuclear capacity for SMR/LMR labels."""
    if "_" not in short_name:
        return None

    reactor_capacity, load_mode = short_name.split("_", 1)
    if load_mode != "BA":
        return None

    reactor = reactor_capacity[:3]
    if reactor not in {"SMR", "LMR"}:
        return None

    unit_capacity_mw = int(reactor_capacity[3:])
    multiplier = 5 if reactor == "SMR" else 2
    return reactor, unit_capacity_mw * multiplier


capacity_price_rows = []

for scenario_name, scenario in scenarios.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)

    parsed = parse_reactor_capacity(short_name)
    if scenario_name == BASELINE_SCENARIO:
        reactor_capacity_pairs = [("SMR", 0), ("LMR", 0)]
        marker_type = "B30"
    elif parsed is not None:
        reactor_capacity_pairs = [parsed]
        marker_type = parsed[0]
    else:
        continue

    try:
        busbar_names = set(scenario.get_busbar_names())
    except Exception as e:
        logger.warning(f"  Skipping {short_name}: failed to read busbars ({e})")
        continue

    area_means = {}
    for area in NO_AREAS:
        if area not in busbar_names:
            continue
        try:
            prices = scenario.get_prices_for_busbar(area).values.flatten()
        except Exception as e:
            logger.warning(f"  Skipping {short_name}/{area}: failed to read prices ({e})")
            continue
        area_means[area] = float(np.mean(prices))

    if not area_means:
        continue

    area_means["NO"] = float(np.mean([area_means[area] for area in NO_AREAS if area in area_means]))

    for reactor, installed_capacity_mw in reactor_capacity_pairs:
        for area in CAPACITY_LINE_AREA_ORDER:
            if area not in area_means:
                continue
            capacity_price_rows.append(
                {
                    "Scenario": short_name,
                    "Reactor": reactor,
                    "Marker": marker_type,
                    "Area": area,
                    "Installed_Nuclear_Capacity_MW": installed_capacity_mw,
                    "Mean_Price_EUR_MWh": area_means[area],
                }
            )

df_capacity_price = pd.DataFrame(capacity_price_rows)
capacity_line_csv = paper_output_path / "mean_power_prices_ba_capacity_line.csv"
df_capacity_price.to_csv(capacity_line_csv, index=False, float_format="%.2f")
logger.info(f"Saved BA mean price capacity data to: {capacity_line_csv}")

if not df_capacity_price.empty:
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    marker_by_reactor = {"SMR": "*", "LMR": "s"}
    linestyle_by_reactor = {"SMR": "-", "LMR": "--"}

    for area in CAPACITY_LINE_AREA_ORDER:
        color = CAPACITY_LINE_AREA_COLORS[area]
        for reactor in ["SMR", "LMR"]:
            plot_df = df_capacity_price[
                (df_capacity_price["Area"] == area)
                & (df_capacity_price["Reactor"] == reactor)
                & (df_capacity_price["Marker"] != "B30")
            ].sort_values("Installed_Nuclear_Capacity_MW")
            if plot_df.empty:
                continue
            ax.plot(
                plot_df["Installed_Nuclear_Capacity_MW"],
                plot_df["Mean_Price_EUR_MWh"],
                color=color,
                linestyle=linestyle_by_reactor[reactor],
                linewidth=1.2 if area != "NO" else 1.8,
                marker=marker_by_reactor[reactor],
                markersize=7 if reactor == "SMR" else 5,
                alpha=0.95,
            )

    baseline_df = df_capacity_price[
        (df_capacity_price["Marker"] == "B30")
        & (df_capacity_price["Reactor"] == "SMR")
    ]
    for area in CAPACITY_LINE_AREA_ORDER:
        row = baseline_df[baseline_df["Area"] == area]
        if row.empty:
            continue
        ax.scatter(
            row["Installed_Nuclear_Capacity_MW"],
            row["Mean_Price_EUR_MWh"],
            color=CAPACITY_LINE_AREA_COLORS[area],
            marker="o",
            s=24 if area != "NO" else 34,
            zorder=5,
        )

    area_handles = [
        Line2D(
            [0],
            [0],
            color=CAPACITY_LINE_AREA_COLORS[area],
            linewidth=1.8 if area == "NO" else 1.2,
            label=area,
        )
        for area in CAPACITY_LINE_AREA_ORDER
    ]
    marker_handles = [
        Line2D([0], [0], color="0.45", marker="*", linestyle="-", markersize=8, label="SMR"),
        Line2D([0], [0], color="0.45", marker="s", linestyle="--", markersize=5, label="LMR"),
        Line2D([0], [0], color="0.45", marker="o", linestyle="None", markersize=5, label="B30"),
    ]

    area_legend = ax.legend(
        handles=area_handles,
        loc="upper center",
        ncol=6,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.47, 1.03),
    )
    ax.add_artist(area_legend)
    ax.legend(handles=marker_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xlabel("Total Installed Nuclear Capacity (MW)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Price (€/MWh)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-150, 8200)
    ax.set_xticks([0, 1500, 3000, 4000, 4500, 6000, 8000])
    ax.tick_params(labelsize=9)

    output_file_capacity_line = paper_output_path / "mean_power_prices_ba_capacity_line.pdf"
    plt.tight_layout()
    plt.savefig(output_file_capacity_line, dpi=300, bbox_inches="tight")
    logger.info(f"Saved BA mean price capacity line plot to: {output_file_capacity_line}")
else:
    output_file_capacity_line = None
    logger.warning("No data available for BA mean price capacity line plot")

logger.info("\n" + "=" * 80)
logger.info("✓ Script completed successfully!")
logger.info("=" * 80)
logger.info("Generated files:")
logger.info(f"  - {output_file_ba.name}")
logger.info(f"  - {output_file_llps.name}")
logger.info(f"  - {output_file_comp.name}")
logger.info(f"  - {output_csv.name}")
logger.info(f"  - {capacity_line_csv.name}")
if output_file_capacity_line is not None:
    logger.info(f"  - {output_file_capacity_line.name}")
logger.info("=" * 80)
