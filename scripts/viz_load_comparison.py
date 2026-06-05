"""
Generate load comparison visualization.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import ScenarioStyler, add_grouped_legend, load_scenarios, logger

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
        "../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/test_group/load_comparison_debug.pdf"
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
logger.info(f"Generating load comparison for {len(scenarios)} scenarios...")

# Find common busbars
all_busbars = [set(scenario.get_busbar_names()) for scenario in scenarios.values()]
common_busbars = sorted(set.intersection(*all_busbars)) if all_busbars else []
busbar_names = [b for b in busbars if b in common_busbars]

if not busbar_names:
    logger.warning("No common busbars found")
    exit(1)

ncols = 2
nrows = (len(busbar_names) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
axes = np.atleast_1d(axes).flatten()

scenario_names_str = ", ".join(scenarios.keys())
fig.suptitle(f"Load Comparison (Mean) - Scenarios: {scenario_names_str}", fontsize=14, fontweight="bold", y=0.995)

for idx, busbar_name in enumerate(busbar_names):
    ax = axes[idx]

    for scenario_name, scenario in scenarios.items():
        try:
            df_load = scenario.get_load_for_busbar(busbar_name)
            mean_load = df_load.mean(axis=1).values

            style = styler.mpl_style(scenario_name)
            ax.plot(
                mean_load,
                label=scenario_name,
                color=style.color,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                marker=style.marker,
                markersize=style.markersize,
                markevery=max(1, len(mean_load) // 20),
                alpha=0.8,
            )
        except Exception as e:
            logger.warning(f"Failed to get load for {busbar_name} in {scenario_name}: {e}")

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Load (MW)")
    ax.set_title(f"Load Profile - {busbar_name}")
    ax.grid(True, alpha=0.3)
    add_grouped_legend(ax, styler)

for idx in range(len(busbar_names), len(axes)):
    axes[idx].axis("off")

fig.tight_layout()
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)

logger.info(f"Saved load comparison to {output_file}")
