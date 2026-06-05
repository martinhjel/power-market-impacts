"""
Plot reservoir trajectories for OWN, OW, and N scenarios using Plotly.
Shows interactive comparison of different technology scenarios.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.common import ScenarioStyler, load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Scenarios from OW_N_OWN group
SCENARIOS = [
    "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF",
    "BASELINE_20TWh_FalseHYD_FalseFF_BALOAD_20.00TWH_NoneNUKE_NoneOFF",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    
    "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
]

# Shorter names for display
SCENARIO_LABELS = {
    "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF": "Baseline",
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF": "Baseline Uprate",
    "BASELINE_20TWh_FalseHYD_FalseFF_BALOAD_20.00TWH_NoneNUKE_NoneOFF": "Baseline20",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "LLPS_N Uprate",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "LLPS_OWN Uprate",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "LLPS_OW Uprate",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "BA_N Uprate",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "BA_OWN Uprate",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "BA_OW Uprate",
    
    "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "LLPS_N",
    "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "LLPS_OWN",
    "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "LLPS_OW",
    "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "BA_N",
    "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "BA_OWN",
    "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "BA_OW",
}

# Norwegian busbars to sum
NO_BUSBARS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)

# Load scenarios
scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

logger.info(f"Loaded {len(scenarios)} scenarios")

# Load historical reservoir data
historical_data = None
try:
    historical_data_path = base_path / "app/data/historic_reservoir_nve.parquet"
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

            historical_data = hist_weekly_stats
            logger.info("Loaded historical reservoir data from NVE")
    else:
        logger.warning(f"Historical data file not found at {historical_data_path}")
except Exception as e:
    logger.warning(f"Failed to load historical reservoir data: {e}")

# Collect reservoir data for each scenario
scenario_reservoir_data = {}

for scenario_name, scenario in scenarios.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)

    total_reservoir = None
    max_volume = 0.0

    for area in NO_BUSBARS:
        try:
            val = scenario.get_reservoir_for_busbar(area)
            total_reservoir = val if total_reservoir is None else total_reservoir + val
            max_volume += val.max().max()

        except Exception as e:
            logger.warning(f"Failed to get reservoir for {area} in {scenario_name}: {e}")

    if total_reservoir is not None and max_volume > 0:
        # Normalize to percentage (0-100)
        total_reservoir_pct = (total_reservoir / max_volume) * 100

        # Calculate statistics across scenarios (columns are different historical scenarios)
        mean_trajectory = total_reservoir_pct.mean(axis=1)
        median_trajectory = total_reservoir_pct.quantile(0.5, axis=1)
        p10_trajectory = total_reservoir_pct.quantile(0.1, axis=1)
        p90_trajectory = total_reservoir_pct.quantile(0.9, axis=1)

        scenario_reservoir_data[short_name] = {
            "mean": mean_trajectory,
            "median": median_trajectory,
            "p10": p10_trajectory,
            "p90": p90_trajectory,
            "time_steps": np.arange(len(mean_trajectory)),
        }

        logger.info(f"Processed reservoir data for {short_name}")
    else:
        logger.warning(f"No reservoir data for {scenario_name}")

# Initialize styler for consistent colors
styler = ScenarioStyler()

# Create Plotly figure
fig = go.Figure()

# Add historical data first (if available) so it's in the background
if historical_data is not None and len(scenario_reservoir_data) > 0:
    # Get the length from the first scenario
    first_scenario = list(scenario_reservoir_data.values())[0]
    num_steps = len(first_scenario["mean"])
    time_steps = np.arange(num_steps)

    # Tile historical data to match simulation length
    hist_mean = np.tile(historical_data["mean"].values * 100, (num_steps // 52) + 1)[:num_steps]
    hist_p10 = np.tile(historical_data["p10"].values * 100, (num_steps // 52) + 1)[:num_steps]
    hist_p50 = np.tile(historical_data["p50"].values * 100, (num_steps // 52) + 1)[:num_steps]
    hist_p90 = np.tile(historical_data["p90"].values * 100, (num_steps // 52) + 1)[:num_steps]

    # Add historical p10-p90 range
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=hist_p90,
            mode="lines",
            name="Historical p90",
            line=dict(color="rgba(128,128,128,0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=hist_p10,
            mode="lines",
            name="Historical p10-p90",
            line=dict(color="rgba(128,128,128,0)", width=0),
            fillcolor="rgba(128,128,128,0.2)",
            fill="tonexty",
            showlegend=True,
            hovertemplate="Historical p10-p90<br>Week: %{x}<br>Fill: %{y:.1f}%<extra></extra>",
        )
    )

    # Add historical mean
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=hist_mean,
            mode="lines",
            name="Historical Mean",
            line=dict(color="black", width=2.5, dash="solid"),
            opacity=0.7,
            hovertemplate="Historical Mean<br>Week: %{x}<br>Fill: %{y:.1f}%<extra></extra>",
        )
    )

    # Add historical median
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=hist_p50,
            mode="lines",
            name="Historical Median",
            line=dict(color="darkgray", width=2.0, dash="dash"),
            opacity=0.6,
            hovertemplate="Historical Median<br>Week: %{x}<br>Fill: %{y:.1f}%<extra></extra>",
        )
    )

# Add scenario data
for short_name, data in scenario_reservoir_data.items():
    # Get color from styler
    style = styler.mpl_style(short_name)

    # Convert matplotlib color to plotly color
    color = style.color

    # Add p10-p90 range as filled area
    fig.add_trace(
        go.Scatter(
            x=data["time_steps"],
            y=data["p90"],
            mode="lines",
            name="p90",
            legendgroup=short_name,
            legendgrouptitle_text=short_name,
            line=dict(color=color, width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["time_steps"],
            y=data["p10"],
            mode="lines",
            name="p10-p90",
            legendgroup=short_name,
            legendgrouptitle_text=short_name,
            line=dict(color=color, width=0),
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)",
            fill="tonexty",
            showlegend=True,
            hovertemplate=f"{short_name} p10-p90<br>Week: %{{x}}<br>Fill: %{{y:.1f}}%<extra></extra>",
        )
    )

    # Add mean trajectory
    fig.add_trace(
        go.Scatter(
            x=data["time_steps"],
            y=data["mean"],
            mode="lines",
            name="Mean",
            legendgroup=short_name,
            line=dict(color=color, width=3),
            hovertemplate=f"{short_name} Mean<br>Week: %{{x}}<br>Fill: %{{y:.1f}}%<extra></extra>",
        )
    )

    # Add median trajectory (dashed)
    fig.add_trace(
        go.Scatter(
            x=data["time_steps"],
            y=data["median"],
            mode="lines",
            name="Median",
            legendgroup=short_name,
            line=dict(color=color, width=2, dash="dash"),
            opacity=0.7,
            hovertemplate=f"{short_name} Median<br>Week: %{{x}}<br>Fill: %{{y:.1f}}%<extra></extra>",
        )
    )

# Update layout
fig.update_layout(
    title=dict(
        text="Norway Total Reservoir Trajectory - OW_N_OWN Scenarios",
        font=dict(size=18, family="Arial", color="black"),
        x=0.5,
        xanchor="center",
    ),
    xaxis=dict(
        title="Week",
        title_font=dict(size=14, family="Arial"),
        tickfont=dict(size=12, family="Arial"),
        gridcolor="lightgray",
        showgrid=True,
    ),
    yaxis=dict(
        title="Total Reservoir Filling (%)",
        title_font=dict(size=14, family="Arial"),
        tickfont=dict(size=12, family="Arial"),
        range=[0, 100],
        gridcolor="lightgray",
        showgrid=True,
    ),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1.0,
        xanchor="left",
        x=1.02,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
    ),
    width=1000,
    height=600,
)

# Save the figure
output_file_html = paper_output_path / "reservoir_trajectory_scenarios.html"
output_file_pdf = paper_output_path / "reservoir_trajectory_scenarios.pdf"

fig.write_html(output_file_html)
logger.info(f"Saved interactive reservoir trajectory to {output_file_html}")

# Try to save as PDF (requires kaleido)
try:
    fig.write_image(output_file_pdf, width=800, height=800, scale=2)
    logger.info(f"Saved reservoir trajectory to {output_file_pdf}")
except Exception as e:
    logger.warning(f"Could not save PDF (kaleido may not be installed): {e}")

print("\nVisualizations saved to:")
print(f"  HTML: {output_file_html}")
if output_file_pdf.exists():
    print(f"  PDF: {output_file_pdf}")
