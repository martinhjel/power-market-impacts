"""
Generate price statistics comparison (bar charts).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import ScenarioStyler, load_scenarios, logger

if "snakemake" not in dir():

    class DebugConfig:
        class input:
            metadata = "../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/scenario_metadata.json"

        class output:
            pass

        class params:
            busbars = ["NO1", "NO2", "NO3"]
            scenarios = [
                "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
                "BASELINE_10TWh_FalseHYD_FalseFF_BALOAD_10.00TWH_NoneNUKE_NoneOFF",
            ]

        class wildcards:
            group = "test_group"

    DebugConfig.output = [
        "../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/test_group/price_statistics_debug.pdf"
    ]
    snakemake = DebugConfig

# Snakemake inputs/outputs
metadata_file = Path(snakemake.input.metadata)
output_file = Path(snakemake.output[0])
busbars = snakemake.params.busbars
group_scenarios = snakemake.params.scenarios

# Load metadata
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Load only the scenarios for this group
scenario_paths = {name: Path(path) for name, path in metadata["scenarios"].items() if name in group_scenarios}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

# Create output directory
output_file.parent.mkdir(parents=True, exist_ok=True)

# Initialize styler
styler = ScenarioStyler()

# Generate visualization
logger.info(f"Generating price statistics for {len(scenarios)} scenarios...")

# Find common busbars
all_busbars = [set(scenario.get_busbar_names()) for scenario in scenarios.values()]
common_busbars = sorted(set.intersection(*all_busbars)) if all_busbars else []
busbar_names = [b for b in busbars if b in common_busbars]

if not busbar_names:
    logger.warning("No common busbars found")
    exit(1)

# Calculate statistics
stats_data = {}
for scenario_name, scenario in scenarios.items():
    stats_data[scenario_name] = {}
    for busbar_name in busbar_names:
        try:
            df_prices = scenario.get_prices_for_busbar(busbar_name)
            stats_data[scenario_name][busbar_name] = {
                "mean": df_prices.mean().mean(),
                "median": df_prices.median().median(),
                "std": df_prices.std().std(),
                "p10": df_prices.quantile(0.1).quantile(0.1),
                "p90": df_prices.quantile(0.9).quantile(0.9),
            }
        except Exception as e:
            logger.warning(f"Failed to get prices for {busbar_name} in {scenario_name}: {e}")

# Create bar charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

scenario_names_str = ", ".join(scenarios.keys())
fig.suptitle(f"Price Statistics Comparison - Scenarios: {scenario_names_str}", fontsize=14, fontweight="bold", y=0.995)

stats_to_plot = ["mean", "median", "std", "p90"]
stat_labels = ["Mean Price (€/MWh)", "Median Price (€/MWh)", "Std Dev (€/MWh)", "90th Percentile (€/MWh)"]

for idx, (stat, label) in enumerate(zip(stats_to_plot, stat_labels)):
    ax = axes[idx]

    x = np.arange(len(busbar_names))
    width = 0.8 / len(scenarios)

    for i, (scenario_name, scenario_stats) in enumerate(stats_data.items()):
        values = [scenario_stats.get(b, {}).get(stat, 0) for b in busbar_names]
        style = styler.mpl_style(scenario_name)
        offset = (i - len(scenarios) / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=scenario_name, color=style.color, alpha=0.8)

    ax.set_xlabel("Busbar")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels(busbar_names, rotation=45, ha="right")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)

logger.info(f"Saved price statistics to {output_file}")
