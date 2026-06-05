#!/usr/bin/env python3
"""
Plot offshore wind, nuclear, and load profiles in MW with correlation analysis.

This script visualizes the daily average generation/load profiles for:
- Offshore wind sites (Sørlige Nordsjø II, Utsira Nord, Vestavind D) - individual and total
- Nuclear sites (combined profile)
- Norwegian load (NO1-NO5 total)

All profiles are scaled appropriately and correlations are computed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import ScenarioResults

# Offshore wind configurations with capacities
WIND_CONFIGS = {
    "NO2_wind_offshore_SorligeNordsjo2": {
        "capacity": 3000,
        "name": "Sørlige Nordsjø II (3000 MW)",
        "color": "steelblue",
    },
    "NO2_wind_offshore_UtsiraNord": {"capacity": 500, "name": "Utsira Nord (500 MW)", "color": "royalblue"},
    "NO5_wind_offshore_Vestavind_D": {"capacity": 1500, "name": "Vestavind D (1500 MW)", "color": "navy"},
}

# Nuclear configuration (total capacity)
NUCLEAR_CAPACITY = 900.84 + 2005.10  # NO2 + NO1 = 2905.94 MW

# Configuration for loading scenario data
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
BASELINE_SCENARIO = "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF"
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]


def load_norwegian_load():
    """Load total Norwegian load from baseline scenario."""
    base_path = Path.cwd()
    result_path = base_path / "ltm_output" / MODEL_FOLDER / BASELINE_SCENARIO

    print(f"Loading load data from {BASELINE_SCENARIO}...")
    scenario = ScenarioResults(result_path)

    # Sum load across all Norwegian areas
    total_load = None
    for area in NO_AREAS:
        try:
            load_df = scenario.get_load_for_busbar(area)
        except KeyError:
            continue
        if total_load is None:
            total_load = load_df
        else:
            total_load = total_load + load_df

    # Average across scenarios (columns) and convert to Series
    if total_load is not None:
        total_load_series = total_load.mean(axis=1)
        return total_load_series
    return None


def main():
    print("Loading renewable profiles...")
    df_renewables = pd.read_parquet("data/renewables_profiles.parquet")

    print("Loading nuclear profiles...")
    df_nuclear = pd.read_parquet("data/new_nuclear_profile.parquet")

    # Load Norwegian load
    load_series = load_norwegian_load()

    # Check which wind profiles are available
    available_wind = [p for p in WIND_CONFIGS.keys() if p in df_renewables.columns]
    print(f"Available wind profiles: {available_wind}")

    # Create plot with secondary y-axis for load
    print("\nCreating plot...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()  # Secondary y-axis for load

    # Storage for correlation analysis
    profile_data_for_corr = {}

    # Plot nuclear profile (scaled by capacity)
    df_nuclear_daily = df_nuclear.resample("D").mean()
    df_nuclear_daily["day"] = df_nuclear_daily.index.dayofyear
    nuclear_mw = df_nuclear_daily["capacity_factor"].values.flatten() * NUCLEAR_CAPACITY

    ax1.plot(
        df_nuclear_daily["day"].values,
        nuclear_mw,
        linewidth=2.5,
        label=f"Nuclear Total ({NUCLEAR_CAPACITY:.0f} MW)",
        alpha=0.8,
        color="red",
        linestyle="--",
    )

    # Store hourly nuclear for correlation
    nuclear_hourly = (df_nuclear["capacity_factor"] * NUCLEAR_CAPACITY).values.flatten()
    profile_data_for_corr["Nuclear"] = nuclear_hourly

    # Track total wind generation
    wind_total = None
    wind_total_hourly = None

    # Plot individual wind profiles (scaled by capacity)
    for profile_name in available_wind:
        config = WIND_CONFIGS[profile_name]
        profile_data = df_renewables[profile_name]

        # Daily for plotting
        ind_daily = profile_data.index.dayofyear
        df_daily = pd.DataFrame({"value": profile_data.values}, index=ind_daily).groupby(level=0).mean()
        generation_mw = df_daily["value"].values * config["capacity"]

        # Hourly for correlation
        hourly_mw = profile_data.values * config["capacity"]
        profile_data_for_corr[config["name"].split("(")[0].strip()] = hourly_mw

        # Add to total
        if wind_total is None:
            wind_total = generation_mw
            wind_total_hourly = hourly_mw
        else:
            wind_total = wind_total + generation_mw
            wind_total_hourly = wind_total_hourly + hourly_mw

        ax1.plot(df_daily.index, generation_mw, linewidth=1.5, label=config["name"], alpha=0.7, color=config["color"])

    # Store wind total for correlation
    if wind_total_hourly is not None:
        profile_data_for_corr["Offshore Wind Total"] = wind_total_hourly

    # Plot total offshore wind
    if wind_total is not None:
        total_capacity = sum(cfg["capacity"] for cfg in WIND_CONFIGS.values())
        days = df_daily.index
        ax1.plot(
            days,
            wind_total,
            linewidth=2.5,
            label=f"Offshore Wind Total ({total_capacity:.0f} MW)",
            alpha=0.9,
            color="darkblue",
            linestyle="-",
        )

    # Plot load on secondary axis
    if load_series is not None:
        # Convert to daily averages
        load_hourly = load_series.values
        ind_daily = load_series.index.dayofyear
        df_load_daily = pd.DataFrame({"value": load_hourly}, index=ind_daily).groupby(level=0).mean()

        ax2.plot(
            df_load_daily.index,
            df_load_daily["value"],
            linewidth=2.5,
            label="Norwegian Load",
            alpha=0.8,
            color="orange",
            linestyle=":",
        )

        # Store load for correlation
        profile_data_for_corr["Load"] = load_hourly

    # Formatting
    ax1.set_xlabel("Day of Year", fontsize=12)
    ax1.set_ylabel("Average Daily Generation (MW)", fontsize=12, color="black")
    ax2.set_ylabel("Average Daily Load (MW)", fontsize=12, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax1.set_title("Offshore Wind, Nuclear, and Load Profiles", fontsize=14, fontweight="bold")
    ax1.set_xlim(1, 365)
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = Path("visualizations/paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "generation_profiles_mw.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Calculate correlations
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS (Hourly Data)")
    print("=" * 70)

    # Ensure all arrays have the same length by trimming to minimum
    min_length = min(len(v) for v in profile_data_for_corr.values())
    for key in profile_data_for_corr:
        profile_data_for_corr[key] = profile_data_for_corr[key][:min_length]

    # Create correlation matrix
    profile_names = list(profile_data_for_corr.keys())
    n = len(profile_names)
    corr_matrix = np.zeros((n, n))

    for i, name1 in enumerate(profile_names):
        for j, name2 in enumerate(profile_names):
            corr_matrix[i, j] = np.corrcoef(profile_data_for_corr[name1], profile_data_for_corr[name2])[0, 1]

    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print("-" * 70)

    # Header
    header = "".ljust(25) + " ".join(f"{i + 1:>6}" for i in range(n))
    print(header)
    print("-" * 70)

    # Rows
    for i, name1 in enumerate(profile_names):
        row = f"{i + 1}. {name1}".ljust(25) + " ".join(f"{corr_matrix[i, j]:>6.3f}" for j in range(n))
        print(row)

    print("\nProfile Legend:")
    for i, name in enumerate(profile_names):
        print(f"{i + 1}. {name}")

    # Key correlations
    print("\n" + "=" * 70)
    print("KEY CORRELATIONS")
    print("=" * 70)

    # Wind-Nuclear correlations
    if "Offshore Wind Total" in profile_data_for_corr and "Nuclear" in profile_data_for_corr:
        wind_nuclear_corr = np.corrcoef(profile_data_for_corr["Offshore Wind Total"], profile_data_for_corr["Nuclear"])[
            0, 1
        ]
        print(f"Offshore Wind Total vs Nuclear: {wind_nuclear_corr:>8.3f}")

    # Wind-Load correlations
    if "Offshore Wind Total" in profile_data_for_corr and "Load" in profile_data_for_corr:
        wind_load_corr = np.corrcoef(profile_data_for_corr["Offshore Wind Total"], profile_data_for_corr["Load"])[0, 1]
        print(f"Offshore Wind Total vs Load:    {wind_load_corr:>8.3f}")

    # Nuclear-Load correlation
    if "Nuclear" in profile_data_for_corr and "Load" in profile_data_for_corr:
        nuclear_load_corr = np.corrcoef(profile_data_for_corr["Nuclear"], profile_data_for_corr["Load"])[0, 1]
        print(f"Nuclear vs Load:                 {nuclear_load_corr:>8.3f}")

    # Individual wind farm vs Load
    print("\nIndividual Wind Farms vs Load:")
    for name in ["Sørlige Nordsjø II", "Utsira Nord", "Vestavind D"]:
        if name in profile_data_for_corr and "Load" in profile_data_for_corr:
            corr = np.corrcoef(profile_data_for_corr[name], profile_data_for_corr["Load"])[0, 1]
            print(f"{name:30s} vs Load: {corr:>8.3f}")

    plt.show()


if __name__ == "__main__":
    main()
