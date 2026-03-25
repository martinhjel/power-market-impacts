#!/usr/bin/env python3
"""
Plot offshore wind and nuclear generation profiles in MW.

This script visualizes the daily average generation profiles for:
- Offshore wind sites (Sørlige Nordsjø II, Utsira Nord, Vestavind D) - individual and total
- Nuclear sites (combined profile)

All profiles are scaled by their installed capacities to show MW output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

print("Loading renewable profiles...")
df_renewables = pd.read_parquet("data/renewables_profiles.parquet")

print("Loading nuclear profiles...")
df_nuclear = pd.read_parquet("data/new_nuclear_profile.parquet")

# Check which wind profiles are available
available_wind = [p for p in WIND_CONFIGS.keys() if p in df_renewables.columns]
print(f"Available wind profiles: {available_wind}")

# Create plot
print("\nCreating plot...")
fig, ax = plt.subplots(figsize=(14, 7))

# Plot nuclear profile (scaled by capacity)
df_nuclear_daily = df_nuclear.resample("D").mean()
df_nuclear_daily["day"] = df_nuclear_daily.index.dayofyear
nuclear_mw = df_nuclear_daily["capacity_factor"].values.flatten() * NUCLEAR_CAPACITY

ax.plot(
    df_nuclear_daily["day"].values,
    nuclear_mw,
    linewidth=2.5,
    label=f"Nuclear Total ({NUCLEAR_CAPACITY:.0f} MW)",
    alpha=0.8,
    color="red",
    linestyle="-",
)

# Track total wind generation
wind_total = None

# Plot individual wind profiles (scaled by capacity)
for profile_name in available_wind:
    config = WIND_CONFIGS[profile_name]
    profile_data = df_renewables[profile_name]

    ind_daily = profile_data.index.dayofyear
    df_daily = pd.DataFrame({"value": profile_data.values}, index=ind_daily).groupby(level=0).mean()

    # Ensure only 365 days (remove day 366 if present)
    if len(df_daily) > 365:
        df_daily = df_daily.iloc[:365]

    # Scale by capacity to get MW
    generation_mw = df_daily["value"].values * config["capacity"]

    # Add to total
    if wind_total is None:
        wind_total = generation_mw
    else:
        wind_total = wind_total + generation_mw

    ax.plot(df_daily.index, generation_mw, linewidth=1.5, label=config["name"], alpha=0.7, color=config["color"])

# Plot total offshore wind
if wind_total is not None:
    total_capacity = sum(cfg["capacity"] for cfg in WIND_CONFIGS.values())
    days = df_daily.index
    ax.plot(
        days,
        wind_total,
        linewidth=2.5,
        label=f"Offshore Wind Total ({total_capacity:.0f} MW)",
        alpha=0.9,
        color="darkblue",
        linestyle="-",
    )

# Plot load
import numpy as np
from lpr_sintef_bifrost.utils.time import CET_winter

from data import PowerGamaDataLoader

dataset_year = 2025
dataset_scenario = "BM"
dataset_version = "100"
base_path = Path.cwd() / "data/NordicNuclearAnalysis"
combined = True

start_scenario_year = 1991
end_scenario_year = 2020

simulation_start = pd.Timestamp(year=2024, month=1, day=1, hour=0, minute=0, second=0, tz=CET_winter)
simulation_years = 1
simulation_end = simulation_start + pd.Timedelta(weeks=52 * simulation_years)
simulation_time_index = pd.date_range(start=simulation_start, end=simulation_end, freq="1h")

data_loader = PowerGamaDataLoader(
    year=dataset_year, scenario=dataset_scenario, version=dataset_version, base_path=base_path, combined=combined
)

df_load_profiles = pd.read_parquet("data/load_profiles.parquet")

# %% Load data

df_load_norway = None
for area in ["NO1", "NO2", "NO3", "NO4", "NO5"]:
    row = data_loader.consumer.loc[data_loader.consumer["node"] == area].iloc[0]

    df_load = pd.read_csv(f"data/Profiler/Consumption/{area}_consumption.csv", index_col=0, parse_dates=True)
    df_load = df_load.loc[(df_load.index.year >= start_scenario_year) & (df_load.index.year <= end_scenario_year)]

    data = []
    for scenario in range(start_scenario_year, end_scenario_year + 1):
        ind = df_load.index.year >= scenario
        end_ind = len(simulation_time_index)

        data.append(np.squeeze(df_load[ind][:end_ind].values))  # TODO: Handle to start from start if more than n_years

    data = np.array(data)
    df = pd.DataFrame(
        index=simulation_time_index, data=data.T, columns=range(start_scenario_year, end_scenario_year + 1)
    )

    dff = df / df.mean()  # Normalize to average 1.0
    dff = dff * row["demand_avg"]

    if df_load_norway is None:
        df_load_norway = dff.copy()
    else:
        df_load_norway += dff

df_load_daily = df_load_norway.resample("D").mean().mean(axis=1)

# Create second y-axis for load
ax2 = ax.twinx()
ax2.plot(
    range(1, len(df_load_daily) + 1),
    df_load_daily.values,
    linewidth=2.5,
    label=f"Total Load Norway ({df_load_daily.mean():.0f} MW avg)",
    alpha=0.9,
    color="black",
    linestyle="--",
)
ax2.set_ylabel("Average Daily Load (MW)", fontsize=12)
ax2.set_ylim(bottom=0.0)


# Config
ax.set_xlabel("Day of Year", fontsize=12)
ax.set_ylabel("Average Daily Generation (MW)", fontsize=12)
ax.set_title("Offshore Wind and Nuclear Average Daily Generation Profiles", fontsize=14, fontweight="bold")
ax.set_xlim(1, 365)
ax.grid(True, alpha=0.3)

# Combine legends from both axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9, framealpha=0.95)

plt.tight_layout()

# Create output directory if it doesn't exist
output_dir = Path("visualizations/paper")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "generation_profiles_mw.pdf"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to: {output_path}")

plt.show()

# ============================================================================
# CORRELATION ANALYSIS - DAILY AVERAGED PROFILES (SINGLE YEAR AVERAGE)
# ============================================================================

# Compute correlations between profiles averaged across scenarios (365 days)
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS - DAILY AVERAGED PROFILES (SINGLE YEAR AVERAGE)")
print("=" * 80)

# Prepare data for correlation analysis
correlation_data = pd.DataFrame(
    {"Nuclear": nuclear_mw, "Offshore Wind Total": wind_total, "Load": df_load_daily.values}
)

# Add individual wind sites
for profile_name in available_wind:
    config = WIND_CONFIGS[profile_name]
    profile_data = df_renewables[profile_name]
    ind_daily = profile_data.index.dayofyear
    df_daily = pd.DataFrame({"value": profile_data.values}, index=ind_daily).groupby(level=0).mean()

    # Ensure only 365 days (remove day 366 if present)
    if len(df_daily) > 365:
        df_daily = df_daily.iloc[:365]

    generation_mw = df_daily["value"].values * config["capacity"]
    correlation_data[config["name"].split(" (")[0]] = generation_mw

# Compute correlation matrix
corr_matrix = correlation_data.corr()

correlation_data.iloc[:1000, :].plot()

print("\nCorrelation Matrix (365 day averages):")
print(corr_matrix.round(3))

print("\nKey Correlations:")
print(f"  Nuclear vs Load:                {corr_matrix.loc['Nuclear', 'Load']:.3f}")
print(f"  Offshore Wind Total vs Load:    {corr_matrix.loc['Offshore Wind Total', 'Load']:.3f}")
print(f"  Nuclear vs Offshore Wind Total: {corr_matrix.loc['Nuclear', 'Offshore Wind Total']:.3f}")
for profile_name in available_wind:
    site_name = WIND_CONFIGS[profile_name]["name"].split(" (")[0]
    print(f"  {site_name} vs Load: {corr_matrix.loc[site_name, 'Load']:.3f}")

print(f"  Observations: {len(correlation_data)}")
print("=" * 80)

# ============================================================================
# CORRELATION ANALYSIS - MULTIPLE TEMPORAL RESOLUTIONS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS - MULTIPLE TEMPORAL RESOLUTIONS")
print("=" * 80)

# Prepare hourly data across all scenarios
nuclear_hourly = df_nuclear["capacity_factor"].values * NUCLEAR_CAPACITY
nuclear_hourly_all = np.tile(nuclear_hourly, end_scenario_year - start_scenario_year + 1)

wind_hourly_all = None
for profile_name in available_wind:
    config = WIND_CONFIGS[profile_name]
    profile_data = df_renewables[profile_name].values * config["capacity"]
    # Replicate for all scenarios
    profile_all_scenarios = np.tile(profile_data, end_scenario_year - start_scenario_year + 1)
    if wind_hourly_all is None:
        wind_hourly_all = profile_all_scenarios
    else:
        wind_hourly_all = wind_hourly_all + profile_all_scenarios

load_hourly_all = df_load_norway.T.values.flatten()

# Ensure all arrays have the same length
min_length = min(len(nuclear_hourly_all), len(wind_hourly_all), len(load_hourly_all))
nuclear_hourly_all = nuclear_hourly_all[:min_length]
wind_hourly_all = wind_hourly_all[:min_length]
load_hourly_all = load_hourly_all[:min_length]

# Create a time index for resampling
time_index = pd.date_range(start="2024-01-01", periods=min_length, freq="H")

# Create dataframe with hourly data
hourly_data = pd.DataFrame(
    {"Nuclear": nuclear_hourly_all, "Offshore Wind Total": wind_hourly_all, "Load": load_hourly_all}, index=time_index
)

# Calculate correlations at different temporal resolutions
correlations = {}

# 1. Hourly
print("\n1. HOURLY RESOLUTION")
hourly_corr = hourly_data.corr()
correlations["Hourly"] = {
    "Nuclear vs Load": hourly_corr.loc["Nuclear", "Load"],
    "Wind vs Load": hourly_corr.loc["Offshore Wind Total", "Load"],
    "Nuclear vs Wind": hourly_corr.loc["Nuclear", "Offshore Wind Total"],
}
print(f"   Nuclear vs Load:     {correlations['Hourly']['Nuclear vs Load']:.3f}")
print(f"   Wind vs Load:        {correlations['Hourly']['Wind vs Load']:.3f}")
print(f"   Nuclear vs Wind:     {correlations['Hourly']['Nuclear vs Wind']:.3f}")
print(f"   Observations: {len(hourly_data):,}")

# 2. Daily
print("\n2. DAILY RESOLUTION")
daily_data = hourly_data.resample("D").mean()
daily_corr = daily_data.corr()
correlations["Daily"] = {
    "Nuclear vs Load": daily_corr.loc["Nuclear", "Load"],
    "Wind vs Load": daily_corr.loc["Offshore Wind Total", "Load"],
    "Nuclear vs Wind": daily_corr.loc["Nuclear", "Offshore Wind Total"],
}
print(f"   Nuclear vs Load:     {correlations['Daily']['Nuclear vs Load']:.3f}")
print(f"   Wind vs Load:        {correlations['Daily']['Wind vs Load']:.3f}")
print(f"   Nuclear vs Wind:     {correlations['Daily']['Nuclear vs Wind']:.3f}")
print(f"   Observations: {len(daily_data):,}")

# 3. Weekly
print("\n3. WEEKLY RESOLUTION")
weekly_data = hourly_data.resample("W").mean()
weekly_corr = weekly_data.corr()
correlations["Weekly"] = {
    "Nuclear vs Load": weekly_corr.loc["Nuclear", "Load"],
    "Wind vs Load": weekly_corr.loc["Offshore Wind Total", "Load"],
    "Nuclear vs Wind": weekly_corr.loc["Nuclear", "Offshore Wind Total"],
}
print(f"   Nuclear vs Load:     {correlations['Weekly']['Nuclear vs Load']:.3f}")
print(f"   Wind vs Load:        {correlations['Weekly']['Wind vs Load']:.3f}")
print(f"   Nuclear vs Wind:     {correlations['Weekly']['Nuclear vs Wind']:.3f}")
print(f"   Observations: {len(weekly_data):,}")

# 4. Monthly
print("\n4. MONTHLY RESOLUTION")
monthly_data = hourly_data.resample("ME").mean()
monthly_corr = monthly_data.corr()
correlations["Monthly"] = {
    "Nuclear vs Load": monthly_corr.loc["Nuclear", "Load"],
    "Wind vs Load": monthly_corr.loc["Offshore Wind Total", "Load"],
    "Nuclear vs Wind": monthly_corr.loc["Nuclear", "Offshore Wind Total"],
}
print(f"   Nuclear vs Load:     {correlations['Monthly']['Nuclear vs Load']:.3f}")
print(f"   Wind vs Load:        {correlations['Monthly']['Wind vs Load']:.3f}")
print(f"   Nuclear vs Wind:     {correlations['Monthly']['Nuclear vs Wind']:.3f}")
print(f"   Observations: {len(monthly_data):,}")

# ============================================================================
# CREATE LATEX TABLE
# ============================================================================

print("\n" + "=" * 80)
print("LATEX TABLE")
print("=" * 80 + "\n")

latex_table = r"""\begin{table}[htbp]
\centering
\caption{Correlation coefficients between nuclear generation, offshore wind generation, and load at different temporal resolutions (1991--2020 scenarios).}
\label{tab:correlations}
\begin{tabular}{lcccc}
\toprule
\textbf{Pair} & \textbf{Hourly} & \textbf{Daily} & \textbf{Weekly} & \textbf{Monthly} \\
\midrule
"""

# Add rows
pairs = ["Nuclear vs Load", "Wind vs Load", "Nuclear vs Wind"]
for pair in pairs:
    latex_table += f"{pair} & "
    latex_table += f"{correlations['Hourly'][pair]:.3f} & "
    latex_table += f"{correlations['Daily'][pair]:.3f} & "
    latex_table += f"{correlations['Weekly'][pair]:.3f} & "
    latex_table += f"{correlations['Monthly'][pair]:.3f} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex_table)
print("\n" + "=" * 80)

# Copy to clipboard
import pyperclip

try:
    pyperclip.copy(latex_table)
    print("\n✓ LaTeX table copied to clipboard!")
except Exception as e:
    print(f"\n✗ Could not copy to clipboard: {e}")
    print("   Table is printed above for manual copy.")


print("=" * 80 + "\n")

plt.show()

# %%

daily_data.plot(figsize=(14, 6), alpha=0.7)
plt.title("Daily Average Values Across All Scenarios")
plt.ylabel("MW")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
