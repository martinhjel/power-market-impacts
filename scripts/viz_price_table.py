"""
Generate price statistics table with colormap.
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
            busbars = ["NO1", "NO2", "NO3"]
            scenarios = [
                "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
                "BASELINE_10TWh_FalseHYD_FalseFF_BALOAD_10.00TWH_NoneNUKE_NoneOFF",
            ]

        class wildcards:
            group = "test_group"

    DebugConfig.output = ["../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/test_group/price_table_debug.pdf"]
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

# Generate visualization
logger.info(f"Generating price statistics table for {len(scenarios)} scenarios...")

# Find common busbars
all_busbars = [set(scenario.get_busbar_names()) for scenario in scenarios.values()]
common_busbars = sorted(set.intersection(*all_busbars)) if all_busbars else []
busbar_names = [b for b in busbars if b in common_busbars]

if not busbar_names:
    logger.warning("No common busbars found")
    exit(1)

# Calculate statistics
table_data = []
for scenario_name, scenario in scenarios.items():
    for busbar_name in busbar_names:
        try:
            df_prices = scenario.get_prices_for_busbar(busbar_name)
            table_data.append(
                {
                    "Scenario": scenario_name,
                    "Busbar": busbar_name,
                    "Mean": df_prices.mean().mean(),
                    "Median": df_prices.median().median(),
                    "Std": df_prices.std().std(),
                    "Min": df_prices.min().min(),
                    "Max": df_prices.max().max(),
                    "P10": df_prices.quantile(0.1).quantile(0.1),
                    "P90": df_prices.quantile(0.9).quantile(0.9),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to get prices for {busbar_name} in {scenario_name}: {e}")

df_table = pd.DataFrame(table_data)

# Create table visualization
fig, ax = plt.subplots(figsize=(16, max(8, len(table_data) * 0.3)))
ax.axis("tight")
ax.axis("off")

scenario_names_str = ", ".join(scenarios.keys())
fig.suptitle(f"Price Statistics Table - Scenarios: {scenario_names_str}", fontsize=14, fontweight="bold", y=0.98)

# Format numeric columns
numeric_cols = ["Mean", "Median", "Std", "Min", "Max", "P10", "P90"]
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

logger.info(f"Saved price statistics table to {output_file}")
