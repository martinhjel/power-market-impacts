"""
Generate transmission flow duration curves.
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
            scenarios = [
                "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
                "BASELINE_10TWh_FalseHYD_FalseFF_BALOAD_10.00TWH_NoneNUKE_NoneOFF",
            ]

        class wildcards:
            group = "test_group"

    DebugConfig.output = [
        "../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/test_group/transmission_flow_debug.pdf"
    ]
    snakemake = DebugConfig

# Snakemake inputs/outputs
metadata_file = Path(snakemake.input.metadata)
output_file = Path(snakemake.output[0])
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
logger.info(f"Generating transmission flow duration curves for {len(scenarios)} scenarios...")

# Get all DC lines (transmission connections)
all_dclines = []
for scenario in scenarios.values():
    try:
        dclines = scenario.get_dclines()
        for dcline_name in dclines.keys():
            if dcline_name not in all_dclines:
                all_dclines.append(dcline_name)
    except Exception as e:
        logger.warning(f"Failed to get DC lines: {e}")

all_dclines = sorted(all_dclines)

if not all_dclines:
    logger.warning("No DC lines found")
    # Create placeholder
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "No transmission data available", ha="center", va="center", fontsize=14)
    ax.axis("off")
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("No DC lines found, created placeholder")
    exit(0)

ncols = 2
nrows = (len(all_dclines) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
axes = np.atleast_1d(axes).flatten()

scenario_names_str = ", ".join(scenarios.keys())
fig.suptitle(
    f"Transmission Flow Duration Curves - Scenarios: {scenario_names_str}", fontsize=14, fontweight="bold", y=0.995
)

for idx, dcline_name in enumerate(all_dclines):
    ax = axes[idx]

    for scenario_name, scenario in scenarios.items():
        try:
            df_flow = scenario.get_dcline_flow(dcline_name)
            all_flow = df_flow.values.flatten()
            sorted_flow = np.sort(all_flow)[::-1]

            style = styler.mpl_style(scenario_name)
            ax.plot(
                sorted_flow,
                label=scenario_name,
                color=style.color,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                marker=style.marker,
                markersize=style.markersize,
                alpha=0.8,
            )
        except Exception as e:
            logger.warning(f"Failed to get flow for {dcline_name} in {scenario_name}: {e}")

    ax.set_xlabel("Hours (sorted by flow)")
    ax.set_ylabel("Flow (MW)")
    ax.set_title(f"Transmission Flow - {dcline_name}")
    ax.grid(True, alpha=0.3)
    add_grouped_legend(ax, styler)

for idx in range(len(all_dclines), len(axes)):
    axes[idx].axis("off")

fig.tight_layout()
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)

logger.info(f"Saved transmission flow duration curves to {output_file}")
