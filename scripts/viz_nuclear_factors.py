"""
Generate nuclear capacity and value factors visualization.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import ScenarioResults, ScenarioStyler, logger


def get_capacity_value_factor(path, busbar_name, nuke_price):
    """Calculate capacity and value factors for nuclear."""
    scenario = ScenarioResults(Path(path))
    df_price = scenario.get_prices_for_busbar(busbar_name)
    df_load = scenario.get_load_for_busbar(busbar_name)
    try:
        df_nuke_cap = scenario.get_total_nuclear_for_busbar(busbar_name)
        df_nuke_available = scenario.get_total_nuclear_available_for_busbar(busbar_name)
    except KeyError:
        df_nuke_cap = scenario.get_fixed_nuclear_for_busbar(busbar_name)
        df_nuke_available = df_nuke_cap

    if df_nuke_cap.abs().sum().sum() <= 0:
        raise ValueError(f"No processed nuclear generation found for {busbar_name}")

    capacity_factor = df_nuke_cap.mean().mean() / df_nuke_available.max().max()
    achieved_price = (df_price * df_nuke_cap).sum().sum() / df_nuke_cap.sum().sum()
    weighted_price = ((df_price * df_load).sum(axis=1) / df_load.sum(axis=1)).mean()
    value_factor = achieved_price / weighted_price

    return capacity_factor, value_factor, df_nuke_cap


if "snakemake" not in dir():

    class DebugConfig:
        class input:
            metadata = "../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/scenario_metadata.json"

        class output:
            pass

        class params:
            busbars = ["NO1", "NO2", "NO3"]
            nuke_price = 9.0
            scenarios = [
                "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
                "BASELINE_10TWh_FalseHYD_FalseFF_BALOAD_10.00TWH_NoneNUKE_NoneOFF",
            ]

        class wildcards:
            group = "test_group"

    DebugConfig.output = [
        "../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/test_group/nuclear_factors_debug.pdf"
    ]
    snakemake = DebugConfig

# Snakemake inputs/outputs
metadata_file = Path(snakemake.input.metadata)
output_file = Path(snakemake.output[0])
busbars = snakemake.params.busbars
nuke_price = snakemake.params.nuke_price
group_scenarios = snakemake.params.scenarios

# Load metadata
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Load only the scenarios for this group
scenario_paths = {name: Path(path) for name, path in metadata["scenarios"].items() if name in group_scenarios}

# Create output directory
output_file.parent.mkdir(parents=True, exist_ok=True)

# Initialize styler
styler = ScenarioStyler()

# Generate visualization
logger.info(f"Generating nuclear factors for {len(scenario_paths)} scenarios...")

busbar_names = [b for b in busbars if b.startswith("NO")]  # Nuclear typically in Norway

for busbar_name in busbar_names:
    capacity_factors = []
    value_factors = []
    scenario_labels = []
    df_nuke_cap_dict = {}

    for scenario_name, scenario_path in scenario_paths.items():
        try:
            cf, vf, df_nuke_cap = get_capacity_value_factor(scenario_path, busbar_name, nuke_price)
            capacity_factors.append(cf)
            value_factors.append(vf)
            scenario_labels.append(scenario_name)
            df_nuke_cap_dict[scenario_name] = df_nuke_cap
        except ValueError as e:
            logger.debug(f"Skipping {busbar_name} in {scenario_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to calculate factors for {busbar_name} in {scenario_name}: {e}")

    if not capacity_factors:
        logger.info(f"No nuclear data available for {busbar_name}, skipping")
        continue

    # Create bar chart
    x = np.arange(len(scenario_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(scenario_labels) * 0.8), 6))
    bars1 = ax.bar(x - width / 2, capacity_factors, width, label="Capacity Factor", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, value_factors, width, label="Value Factor", color="#ff7f0e")

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Factor")
    ax.set_title(f"Nuclear Capacity and Value Factors - {busbar_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.6, max(max(capacity_factors), max(value_factors)) * 1.05)

    fig.tight_layout()
    factors_output = output_file.parent / f"nuclear_factors_{busbar_name}.pdf"
    fig.savefig(factors_output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved nuclear factors for {busbar_name} to {factors_output}")

    # Duration curve
    fig_duration, ax_duration = plt.subplots(figsize=(12, 6))

    for scenario_name in scenario_labels:
        df_nuke_cap = df_nuke_cap_dict[scenario_name]
        all_values = df_nuke_cap.values.flatten()
        sorted_values = np.sort(all_values)[::-1]
        x_norm = np.linspace(0, 1, len(sorted_values))
        style = styler.mpl_style(scenario_name)

        ax_duration.plot(
            x_norm,
            sorted_values,
            label=scenario_name,
            color=style.color,
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            marker=style.marker,
            markersize=style.markersize,
            markevery=max(1, len(sorted_values) // 20),
            alpha=0.8,
        )

    ax_duration.set_xlabel("Normalized Duration (0 = highest capacity, 1 = lowest)")
    ax_duration.set_ylabel("Nuclear Capacity (MW)")
    ax_duration.set_title(f"Nuclear Capacity Duration Curve - {busbar_name}")
    ax_duration.grid(True, alpha=0.3)
    ax_duration.legend(loc="best", fontsize=8)
    ax_duration.set_xlim(0, 1)

    fig_duration.tight_layout()
    duration_output = output_file.parent / f"nuclear_duration_curve_{busbar_name}.pdf"
    fig_duration.savefig(duration_output, format="pdf", bbox_inches="tight")
    plt.close(fig_duration)

    logger.info(f"Saved nuclear duration curve for {busbar_name} to {duration_output}")

# Create a dummy file if no nuclear data was found
if not Path(output_file).exists():
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "No nuclear data available", ha="center", va="center", fontsize=14)
    ax.axis("off")
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("No nuclear data found, created placeholder file")
