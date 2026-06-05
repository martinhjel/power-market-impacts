"""
Plot average electricity prices on a map for the BASELINE scenario using Plotly.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from scripts.common import load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Scenario to analyze
SCENARIO = "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF"

# Get all available areas from the scenario (will be populated after loading)
AREAS = None  # Will be set to all busbars in the scenario

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)
nodes_file = base_path / "app/data/nodes_location.csv"

# Load node locations
df_nodes = pd.read_csv(nodes_file, index_col="id")
logger.info(f"Loaded {len(df_nodes)} node locations")

# Load scenario
scenario_paths = {SCENARIO: base_path / f"ltm_output/{MODEL_FOLDER}/{SCENARIO}"}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

scenario = scenarios[SCENARIO]
logger.info(f"Loaded scenario: {SCENARIO}")

# Get all busbars from the scenario
AREAS = scenario.get_busbar_names()
logger.info(f"Found {len(AREAS)} areas in scenario: {', '.join(sorted(AREAS))}")

# Create output directory
output_file_html = paper_output_path / "price_map_baseline.html"
output_file_pdf = paper_output_path / "price_map_baseline.pdf"

price_data = {}

for busbar_name in AREAS:
    try:
        df_price = scenario.get_prices_for_busbar(busbar_name)
        df_load = scenario.get_load_for_busbar(busbar_name)

        # Calculate load-weighted price
        weighted_price = ((df_price * df_load).sum(axis=1) / df_load.sum(axis=1)).mean()
        price_data[busbar_name] = weighted_price
        logger.info(f"{busbar_name}: {weighted_price:.2f} €/MWh (load-weighted)")
    except Exception as e:
        logger.warning(f"Failed to get prices for {busbar_name}: {e}")

if not price_data:
    logger.error("No price data collected")
    exit(1)

# Convert to DataFrame for plotting
df_price_map = pd.DataFrame.from_dict(price_data, orient="index", columns=["avg_weighted_price"])
df_price_map.index.name = "id"

# Merge with node coordinates
df_plot = df_nodes.join(df_price_map, how="inner")

# Calculate statistics
min_price = df_plot["avg_weighted_price"].min()
max_price = df_plot["avg_weighted_price"].max()
mean_price = df_plot["avg_weighted_price"].mean()
std_price = df_plot["avg_weighted_price"].std()
min_area = df_plot["avg_weighted_price"].idxmin()
max_area = df_plot["avg_weighted_price"].idxmax()

# Create Plotly figure
fig = go.Figure()

# Add scatter markers for each area
fig.add_trace(
    go.Scattergeo(
        lon=df_plot["lon"],
        lat=df_plot["lat"],
        mode="markers+text",
        marker=dict(
            size=25,
            color=df_plot["avg_weighted_price"],
            colorscale="RdBu_r",  # Red (high) to Blue (low)
            cmin=min_price,
            cmax=max_price,
            colorbar=dict(title="Avg Weighted<br>Price (€/MWh)", thickness=20, len=0.7, x=1.02),
            line=dict(width=2, color="black"),
        ),
        text=[
            f"<b>{node}<br>{price:.1f} €/MWh</b>" for node, price in zip(df_plot.index, df_plot["avg_weighted_price"])
        ],
        textfont=dict(size=10, color="black", family="Arial Black"),
        textposition="top center",
        hovertemplate=("<b>%{text}</b><br>" + "<extra></extra>"),
        showlegend=False,
    )
)

# Update geo layout with higher resolution
fig.update_geos(
    projection_type="mercator",
    resolution=50,  # Higher resolution (default is 110)
    showcountries=True,
    showland=True,
    landcolor="rgb(243, 243, 243)",
    coastlinecolor="rgb(204, 204, 204)",
    countrycolor="rgb(204, 204, 204)",
    showlakes=True,
    lakecolor="rgb(230, 245, 255)",
    showcoastlines=True,
    showocean=True,
    oceancolor="rgb(230, 245, 255)",
    lataxis_range=[50, 72],
    lonaxis_range=[-10, 35],
)

# Update layout
fig.update_layout(
    title=dict(
        text=f"Average Load-Weighted Electricity Prices - All Areas<br><sub>{SCENARIO}</sub>",
        x=0.5,
        xanchor="center",
        font=dict(size=18),
    ),
    margin=dict(l=0, r=100, t=80, b=0),
    width=1400,
    height=900,
    annotations=[
        dict(
            text=(
                "<b>Price Statistics:</b><br>"
                + f"Min: {min_price:.1f} €/MWh ({min_area})<br>"
                + f"Max: {max_price:.1f} €/MWh ({max_area})<br>"
                + f"Mean: {mean_price:.1f} €/MWh<br>"
                + f"Std Dev: {std_price:.1f} €/MWh"
            ),
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11, family="monospace"),
        )
    ],
)

# Save as HTML
fig.write_html(output_file_html)
logger.info(f"Saved interactive price map to {output_file_html}")

# Save as PDF (static image)
try:
    fig.write_image(output_file_pdf, width=1000, height=1200, scale=2)
    logger.info(f"Saved price map to {output_file_pdf}")
except Exception as e:
    logger.warning(f"Could not save PDF (requires kaleido): {e}")
    logger.info("Install kaleido with: pip install kaleido")

print("\nVisualizations saved to:")
print(f"  HTML: {output_file_html}")
if output_file_pdf.exists():
    print(f"  PDF: {output_file_pdf}")
print(f"\nPrice Summary ({len(price_data)} areas):")
for area in sorted(price_data.keys()):
    print(f"  {area}: {price_data[area]:.2f} €/MWh")
print("\nPrice Summary:")
for area in sorted(price_data.keys()):
    print(f"  {area}: {price_data[area]:.2f} €/MWh")
