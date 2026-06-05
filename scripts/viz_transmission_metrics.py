"""
Generate transmission metrics table.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from common import load_scenarios, logger

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
        "../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/test_group/transmission_metrics_debug.pdf"
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

# Generate visualization
logger.info(f"Generating transmission metrics table for {len(scenarios)} scenarios...")

# Get all DC lines
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

# Calculate metrics
table_data = []
for scenario_name, scenario in scenarios.items():
    dclines = scenario.get_dclines()

    for dcline_name in all_dclines:
        if dcline_name in dclines:
            try:
                df_flow = scenario.get_dcline_flow(dcline_name)
                dcline_obj = dclines[dcline_name]

                # Get capacity if available
                capacity = getattr(dcline_obj, "capacity", None)
                if capacity is None:
                    capacity = df_flow.max().max()  # Use max flow as proxy

                table_data.append(
                    {
                        "Scenario": scenario_name,
                        "DC Line": dcline_name,
                        "Mean Flow": df_flow.mean().mean(),
                        "Max Flow": df_flow.max().max(),
                        "Min Flow": df_flow.min().min(),
                        "Utilization %": (df_flow.mean().mean() / capacity * 100) if capacity > 0 else 0,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {dcline_name} in {scenario_name}: {e}")

df_table = pd.DataFrame(table_data)

if df_table.empty:
    # Create placeholder
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "No transmission metrics available", ha="center", va="center", fontsize=14)
    ax.axis("off")
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("No transmission metrics available, created placeholder")
    exit(0)

# Create table visualization
fig, ax = plt.subplots(figsize=(16, max(8, len(table_data) * 0.3)))
ax.axis("tight")
ax.axis("off")

scenario_names_str = ", ".join(scenarios.keys())
fig.suptitle(f"Transmission Flow Metrics - Scenarios: {scenario_names_str}", fontsize=14, fontweight="bold", y=0.98)

# Format numeric columns
numeric_cols = ["Mean Flow", "Max Flow", "Min Flow", "Utilization %"]
for col in numeric_cols:
    df_table[col] = df_table[col].apply(lambda x: f"{x:.2f}")

# Create table
table = ax.table(
    cellText=df_table.values, colLabels=df_table.columns, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

# Style header
for i in range(len(df_table.columns)):
    cell = table[(0, i)]
    cell.set_facecolor("#40466e")
    cell.set_text_props(weight="bold", color="white")

# Alternate row colors
for i in range(1, len(df_table) + 1):
    for j in range(len(df_table.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor("#f0f0f0")
        else:
            cell.set_facecolor("white")

fig.tight_layout()
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)

logger.info(f"Saved transmission metrics table to {output_file}")
