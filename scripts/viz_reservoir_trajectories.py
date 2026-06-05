"""
Generate reservoir trajectories visualization with historical data.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        "../visualizations/PowerGamaMSc_2025_BM_1H_serial_TrueEXO/test_group/reservoir_trajectories_debug.pdf"
    ]
    snakemake = DebugConfig

# Snakemake inputs/outputs
metadata_file = Path(snakemake.input.metadata)
output_file = Path(snakemake.output[0])
busbars = snakemake.params.busbars
group_name = snakemake.wildcards.group
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
logger.info(f"Generating reservoir trajectories for {len(scenarios)} scenarios...")

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
fig.suptitle(
    f"Reservoir Trajectories (Mean + Percentiles) - Scenarios: {scenario_names_str}",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)

for idx, busbar_name in enumerate(busbar_names):
    ax = axes[idx]

    for scenario_name, scenario in scenarios.items():
        try:
            total_reservoir = scenario.get_reservoir_for_busbar(busbar_name)
            max_volume_Mm3 = total_reservoir.max().max()

            if total_reservoir is not None and max_volume_Mm3 > 0:
                total_reservoir_pct = (total_reservoir / max_volume_Mm3) * 100

                mean_reservoir = total_reservoir_pct.mean(axis=1).values
                p10_reservoir = total_reservoir_pct.quantile(0.10, axis=1).values
                p50_reservoir = total_reservoir_pct.quantile(0.50, axis=1).values
                p90_reservoir = total_reservoir_pct.quantile(0.90, axis=1).values

                style = styler.mpl_style(scenario_name)

                ax.plot(
                    mean_reservoir,
                    label=f"{scenario_name} (mean)",
                    color=style.color,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.9,
                )
                ax.plot(
                    p50_reservoir,
                    label=f"{scenario_name} (p50)",
                    color=style.color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax.fill_between(
                    range(len(p10_reservoir)),
                    p10_reservoir,
                    p90_reservoir,
                    color=style.color,
                    alpha=0.2,
                    label=f"{scenario_name} (p10-p90)",
                )

        except Exception as e:
            logger.warning(f"Failed to get reservoir for {busbar_name} in {scenario_name}: {e}")

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Reservoir Filling (%)")
    ax.set_title(f"Reservoir Level Trajectory - {busbar_name}")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)

for idx in range(len(busbar_names), len(axes)):
    axes[idx].axis("off")

fig.tight_layout()
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)

logger.info(f"Saved reservoir trajectories to {output_file}")

# Generate Norway total plot separately
norwegian_busbars = [b for b in busbar_names if b.startswith("NO")]

if norwegian_busbars:
    fig_norway, ax_norway = plt.subplots(figsize=(14, 7))

    scenario_names_str = ", ".join(scenarios.keys())
    fig_norway.suptitle(
        f"Total Norwegian Reservoir Trajectory (Mean + Percentiles) - Scenarios: {scenario_names_str}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for scenario_name, scenario in scenarios.items():
        try:
            total_reservoir = None

            for no_busbar in norwegian_busbars:
                try:
                    val = scenario.get_reservoir_for_busbar(no_busbar)
                    if total_reservoir is None:
                        total_reservoir = val
                    else:
                        total_reservoir = total_reservoir + val
                except Exception as e:
                    logger.warning(f"Failed to get reservoir for {no_busbar} in {scenario_name}: {e}")

            max_volume_Mm3 = total_reservoir.max().max() if total_reservoir is not None else 0.0
            if total_reservoir is not None and max_volume_Mm3 > 0:
                total_reservoir_pct = (total_reservoir / max_volume_Mm3) * 100

                mean_reservoir = total_reservoir_pct.mean(axis=1).values
                p10_reservoir = total_reservoir_pct.quantile(0.10, axis=1).values
                p50_reservoir = total_reservoir_pct.quantile(0.50, axis=1).values
                p90_reservoir = total_reservoir_pct.quantile(0.90, axis=1).values

                style = styler.mpl_style(scenario_name)

                ax_norway.plot(
                    mean_reservoir,
                    label=f"{scenario_name} (mean)",
                    color=style.color,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.9,
                )
                ax_norway.plot(
                    p50_reservoir,
                    label=f"{scenario_name} (p50)",
                    color=style.color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax_norway.fill_between(
                    range(len(p10_reservoir)),
                    p10_reservoir,
                    p90_reservoir,
                    color=style.color,
                    alpha=0.2,
                    label=f"{scenario_name} (p10-p90)",
                )

        except Exception as e:
            logger.warning(f"Failed to calculate total Norwegian reservoir for {scenario_name}: {e}")

    # Add historical data
    try:
        historical_data_path = Path("app/data/historic_reservoir_nve.parquet")
        if historical_data_path.exists():
            df_hist = pd.read_parquet(historical_data_path)
            df_hist_norway = df_hist.loc[df_hist["omrType"] == "NO"]

            if not df_hist_norway.empty:
                df_hist_norway = df_hist_norway.set_index("dato_Id").sort_index()
                df_hist_norway["iso_uke"] = df_hist_norway.index.isocalendar().week

                hist_weekly_stats = df_hist_norway.groupby("iso_uke")["fyllingsgrad"].agg(
                    [
                        ("mean", "mean"),
                        ("p10", lambda x: x.quantile(0.10)),
                        ("p50", lambda x: x.quantile(0.50)),
                        ("p90", lambda x: x.quantile(0.90)),
                    ]
                )

                num_weeks = len(mean_reservoir)
                hist_mean = np.tile(hist_weekly_stats["mean"].values * 100, (num_weeks // 52) + 1)[:num_weeks]
                hist_p10 = np.tile(hist_weekly_stats["p10"].values * 100, (num_weeks // 52) + 1)[:num_weeks]
                hist_p50 = np.tile(hist_weekly_stats["p50"].values * 100, (num_weeks // 52) + 1)[:num_weeks]
                hist_p90 = np.tile(hist_weekly_stats["p90"].values * 100, (num_weeks // 52) + 1)[:num_weeks]

                ax_norway.plot(
                    hist_mean, label="Historical (mean)", color="black", linestyle="-", linewidth=2.5, alpha=0.8
                )
                ax_norway.plot(
                    hist_p50, label="Historical (p50)", color="black", linestyle="--", linewidth=2.0, alpha=0.6
                )
                ax_norway.fill_between(
                    range(len(hist_p10)), hist_p10, hist_p90, color="gray", alpha=0.15, label="Historical (p10-p90)"
                )
    except Exception as e:
        logger.warning(f"Failed to load historical reservoir data: {e}")

    ax_norway.set_xlabel("Time Period")
    ax_norway.set_ylabel("Total Reservoir Filling (%)")
    ax_norway.set_title(f"Total Norwegian Reservoir Level Trajectory (sum of {', '.join(norwegian_busbars)})")
    ax_norway.set_ylim(bottom=0)
    ax_norway.grid(True, alpha=0.3)
    ax_norway.legend(loc="best", fontsize=6, ncol=2)

    fig_norway.tight_layout()
    norway_output = output_file.parent / "reservoir_trajectory_norway_total.pdf"
    fig_norway.savefig(norway_output, format="pdf", bbox_inches="tight")
    plt.close(fig_norway)

    logger.info(f"Saved Norway total reservoir trajectory to {norway_output}")
