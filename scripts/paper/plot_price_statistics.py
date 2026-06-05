"""
Plot price statistics and histograms for OW_N_OWN scenarios.
Shows 10th percentile, mean, median, and 90th percentile with price distributions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from scripts.common import ScenarioStyler, load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Scenarios from OW_N_OWN group
SCENARIOS = [
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
]

# Shorter names for display
SCENARIO_LABELS = {
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "LLPS_N",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "LLPS_OWN",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "LLPS_OW",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "BA_N",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "BA_OWN",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "BA_OW",
}

# Norwegian areas to aggregate
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER


# Load scenarios
scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

logger.info(f"Loaded {len(scenarios)} scenarios")

# Create output directory
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)
output_file_pdf = paper_output_path / "price_distributions_ow_n_own.pdf"
output_file_stats = paper_output_path / "price_statistics_table_ow_n_own.tex"

# Collect price data and calculate statistics for each scenario
statistics = {}
all_prices = {}

for scenario_name, scenario in scenarios.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)

    # Collect prices from all Norwegian areas
    prices_list = []
    for area in NO_AREAS:
        try:
            df_price = scenario.get_prices_for_busbar(area)
            prices_list.append(df_price.values.flatten())
        except Exception as e:
            logger.warning(f"Failed to get prices for {area} in {scenario_name}: {e}")

    if not prices_list:
        logger.warning(f"No price data for {scenario_name}")
        continue

    # Combine all prices
    all_price_values = np.concatenate(prices_list)
    all_prices[short_name] = all_price_values

    # Calculate statistics
    price_stats = {
        "p10": np.percentile(all_price_values, 10),
        "median": np.median(all_price_values),
        "mean": np.mean(all_price_values),
        "p90": np.percentile(all_price_values, 90),
        "std": np.std(all_price_values),
        "min": np.min(all_price_values),
        "max": np.max(all_price_values),
    }
    statistics[short_name] = price_stats

    logger.info(f"{short_name}: Mean={price_stats['mean']:.1f}, Median={price_stats['median']:.1f} €/MWh")

# Initialize styler
styler = ScenarioStyler()

# Create single figure with all distributions
fig, ax = plt.subplots(figsize=(12, 8))

# Determine common x-axis range
all_price_values = np.concatenate([prices for prices in all_prices.values()])
x_min, x_max = np.percentile(all_price_values, [0.5, 99.5])  # Use 99.5th percentile to avoid extreme outliers
x_range = np.linspace(x_min, x_max, 1000)

for short_name, prices in all_prices.items():
    stats_dict = statistics[short_name]

    # Get style
    style = styler.mpl_style(short_name)

    # Fit Gaussian KDE for smooth continuous distribution
    kde = gaussian_kde(prices, bw_method="scott")
    density = kde(x_range)

    # Plot the KDE curve
    ax.plot(
        x_range,
        density,
        linewidth=2.5,
        color=style.color,
        label=f"{short_name} (μ={stats_dict['mean']:.1f})",
        alpha=0.85,
    )

# Format plot
ax.set_xlabel("Price (€/MWh)", fontsize=13, fontweight="bold")
ax.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
ax.set_title("Price Distributions - OW_N_OWN Scenarios (NO1-NO5)", fontsize=15, fontweight="bold", pad=15)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.set_xlim(x_min, x_max)

# Add vertical line at zero for reference if negative prices exist
if x_min < 0:
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

plt.tight_layout()
fig.savefig(output_file_pdf, format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)

logger.info(f"Saved price statistics figure to {output_file_pdf}")

# Generate LaTeX table with statistics
latex_lines = []
latex_lines.append("\\begin{table}[htbp]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Price Statistics for OW\\_N\\_OWN Scenarios (€/MWh)}")
latex_lines.append("\\label{tab:price_stats_ow_n_own}")
latex_lines.append("\\begin{tabular}{lcccccc}")
latex_lines.append("\\hline")
latex_lines.append(
    "\\textbf{Scenario} & \\textbf{Min} & \\textbf{p10} & \\textbf{Median} & \\textbf{Mean} & \\textbf{p90} & \\textbf{Max} \\\\"
)
latex_lines.append("\\hline")

for short_name, price_stats in statistics.items():
    latex_lines.append(
        f"{short_name} & {price_stats['min']:.1f} & {price_stats['p10']:.1f} & "
        f"{price_stats['median']:.1f} & {price_stats['mean']:.1f} & {price_stats['p90']:.1f} & {price_stats['max']:.1f} \\\\"
    )

latex_lines.append("\\hline")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")

# Write to file
with open(output_file_stats, "w") as f:
    f.write("\n".join(latex_lines))

logger.info(f"Saved LaTeX statistics table to {output_file_stats}")

print("\nVisualizations saved to:")
print(f"  PDF: {output_file_pdf}")
print(f"  LaTeX Table: {output_file_stats}")
print("\nPrice Statistics Summary:")
print(f"{'Scenario':<12} {'Mean':>8} {'Median':>8} {'p10':>8} {'p90':>8} {'Std':>8}")
print("-" * 60)
for short_name, price_stats in statistics.items():
    print(
        f"{short_name:<12} {price_stats['mean']:>8.1f} {price_stats['median']:>8.1f} {price_stats['p10']:>8.1f} {price_stats['p90']:>8.1f} {price_stats['std']:>8.1f}"
    )
