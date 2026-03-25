from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lpr_sintef_bifrost.ltm import LTM
from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

from app.utils.market_steps import get_market_steps

st.set_page_config(layout="wide")
st.title("📊 EMPS Simulation Comparison")

###

# result_paths = sorted([i for i in (Path.cwd() / "ltm_output").glob("*/*") if i.is_dir()], key=lambda p: p.name.lower())


def _result_label(p: Path) -> str:
    """Return a short label for a result path: <folder_name> (1H|1D)."""
    parent = p.parent.name
    if "1H" in parent:
        res = "1H"
    elif "1D" in parent:
        res = "1D"
    else:
        # Fallback to parent name if resolution token not found
        res = parent
    return f"{p.name} ({res})"


# path = st.sidebar.selectbox("Select results: ", result_paths, format_func=_result_label)
# st.sidebar.markdown(f"Using path: {path}")

###
# --- Select two simulation folders ---
paths = sorted([i for i in (Path.cwd() / "ltm_output").glob("*/*") if i.is_dir()], key=lambda p: p.name.lower())

paths_dict = {str(i).split("/ltm_output/")[-1]: i for i in paths}

col1, col2 = st.columns(2)
with col1:
    sim1_path = st.selectbox("Select first simulation:", paths, format_func=_result_label, index=0, key="sim1")
    st.sidebar.write("**Sim1**:")
    st.sidebar.write(f"{_result_label(sim1_path)}")

with col2:
    sim2_path = st.selectbox("Select second simulation:", paths, format_func=_result_label, index=1, key="sim2")
    st.sidebar.write("**Sim2**:")
    st.sidebar.write(f"{_result_label(sim2_path)}")


@st.cache_resource
def load_model(path: Path):
    session = LTM.session_from_folder(path / "run_folder/emps")
    return session.model


model1 = load_model(sim1_path)
model2 = load_model(sim2_path)

busbars1 = {b.name: b for b in model1.busbars()}
busbars2 = {b.name: b for b in model2.busbars()}

nodes_csv = Path.cwd() / "app/data/nodes_location.csv"
df_nodes = pd.read_csv(nodes_csv, index_col="id")


def compute_avg_weighted_prices(busbars: Dict[str, Any]) -> Dict[str, float]:
    prices = {}
    for name, busbar in busbars.items():
        df_price = df_from_pyltm_result(busbar.market_result_price())
        df_load = df_from_pyltm_result(busbar.sum_load())
        load_sum = df_load.sum(axis=1)
        valid = load_sum != 0
        if valid.any():
            weighted = (df_price.mul(df_load, axis=0).sum(axis=1)[valid] / load_sum[valid]).mean()
            prices[name] = float(weighted)
        else:
            prices[name] = float("nan")
    return prices


price_map_1 = compute_avg_weighted_prices(busbars1)
price_map_2 = compute_avg_weighted_prices(busbars2)
common_price_nodes = sorted(set(price_map_1).intersection(price_map_2))

df_price_compare = pd.DataFrame(
    {
        "avg_weighted_price_sim1": [price_map_1[node] for node in common_price_nodes],
        "avg_weighted_price_sim2": [price_map_2[node] for node in common_price_nodes],
    },
    index=common_price_nodes,
)
df_price_compare["delta_price"] = (
    df_price_compare["avg_weighted_price_sim2"] - df_price_compare["avg_weighted_price_sim1"]
)

df_price_nodes = df_nodes.join(df_price_compare, how="inner")

st.markdown("## 📋 Price Difference Table")
if df_price_nodes.empty:
    st.info("No overlapping nodes found between the selected simulations.")
else:
    diff_table = df_price_compare.reset_index().rename(
        columns={
            "index": "Node",
            "avg_weighted_price_sim1": "Sim1 Avg Weighted Price",
            "avg_weighted_price_sim2": "Sim2 Avg Weighted Price",
            "delta_price": "Δ Price (Sim2 - Sim1)",
        }
    )
    delta_max = diff_table["Δ Price (Sim2 - Sim1)"].abs().max()
    style = diff_table.style.format(
        {
            "Sim1 Avg Weighted Price": "{:.2f}",
            "Sim2 Avg Weighted Price": "{:.2f}",
            "Δ Price (Sim2 - Sim1)": "{:.2f}",
        }
    )
    if pd.notna(delta_max) and delta_max > 0:
        style = style.background_gradient(
            cmap="RdBu_r",
            subset=["Δ Price (Sim2 - Sim1)"],
            vmin=-delta_max,
            vmax=delta_max,
        )
    st.dataframe(style, hide_index=True)

    st.markdown("## 🗺️ Price Map Comparison")
    diff_fig = go.Figure()
    diff_fig.add_trace(
        go.Scattergeo(
            lon=df_price_nodes["lon"],
            lat=df_price_nodes["lat"],
            mode="markers+text",
            text=[
                f"<b>{node}</b><br>Sim1: {row['avg_weighted_price_sim1']:.1f} €/MWh"
                f"<br>Sim2: {row['avg_weighted_price_sim2']:.1f} €/MWh"
                f"<br>Δ: {row['delta_price']:.1f} €/MWh"
                for node, row in df_price_nodes.iterrows()
            ],
            textposition="top center",
            marker=dict(
                size=12,
                color=df_price_nodes["delta_price"],
                colorscale="RdBu_r",
                cmid=0,
                colorbar=dict(title="Δ Price (Sim2 - Sim1)"),
                line=dict(color="black", width=0.5),
            ),
            hoverinfo="text",
            showlegend=False,
        )
    )
    diff_fig.update_geos(
        projection_type="mercator",
        showcountries=True,
        showland=True,
        landcolor="rgb(240,240,240)",
        lataxis_range=[50, 72],
        lonaxis_range=[-5, 35],
    )
    diff_fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title_text="Average Weighted Price Difference (Sim2 - Sim1)",
        width=1200,
        height=600,
    )
    st.plotly_chart(diff_fig, use_container_width=True)

common_busbars = sorted(set(busbars1).union(busbars2))
default_index = common_busbars.index("NO2") if "NO2" in common_busbars else 0
busbar_name = st.selectbox("Select busbar to compare:", common_busbars, index=default_index)
b1, b2 = busbars1[busbar_name], busbars2[busbar_name]

# Checkbox to toggle between averaged and flattened results
use_flattened = st.checkbox("Do not average out the results (use all per-run values)", value=False)

# Load data for generation mix (needed early)
df_load_1 = df_from_pyltm_result(b1.sum_load())
df_load_2 = df_from_pyltm_result(b2.sum_load())

# --- Generation Mix and Net Import/Export ---
st.header(f"⚡ Generation Mix and Net Import/Export for {busbar_name}")

# Scenario selection and display options
col_scenario1, col_scenario2, col_display = st.columns(3)
with col_scenario1:
    scenario_1 = st.selectbox("Scenario for Sim 1:", ["Mean"] + list(range(df_load_1.shape[1])), key="scenario_1")
with col_scenario2:
    scenario_2 = st.selectbox("Scenario for Sim 2:", ["Mean"] + list(range(df_load_2.shape[1])), key="scenario_2")
with col_display:
    show_individual_lines = st.checkbox("Show individual DC lines", value=False, key="compare_show_individual")


def get_renewables_data(model, busbar_name):
    """Get solar and wind data for a busbar using the proper API.

    Returns: dict with 'solar', 'onshore_wind', 'offshore_wind' DataFrames
    """
    renewables = {"solar": None, "onshore_wind": None, "offshore_wind": None}

    try:
        # Get solar data
        solar_dict = {str(i).split("_")[1]: i for i in model.solar()}
        if busbar_name in solar_dict:
            data = solar_dict[busbar_name].capacity.scenarios
            ind = solar_dict[busbar_name].capacity.timestamps
            renewables["solar"] = pd.DataFrame(data, columns=ind).T
    except Exception:
        pass

    try:
        # Get wind data
        off_wind_dict = {}
        on_wind_dict = {}
        for w in model.wind():
            if "_off" in str(w):
                off_wind_dict[str(w).split("_")[1]] = w
            elif "_on" in str(w):
                on_wind_dict[str(w).split("_")[1]] = w

        # Onshore wind
        if busbar_name in on_wind_dict:
            data = on_wind_dict[busbar_name].capacity.scenarios
            ind = on_wind_dict[busbar_name].capacity.timestamps
            renewables["onshore_wind"] = pd.DataFrame(data, columns=ind).T

        # Offshore wind
        if busbar_name in off_wind_dict:
            data = off_wind_dict[busbar_name].capacity.scenarios
            ind = off_wind_dict[busbar_name].capacity.timestamps
            renewables["offshore_wind"] = pd.DataFrame(data, columns=ind).T
    except Exception:
        pass

    return renewables


def get_connected_dclines(model, busbar_name, df_load_index, scenario):
    """Get individual DC line flows for a busbar.

    Flow direction: positive df_line values = flow from node_a to node_b
    where node names come from dcline name format: "dcline_<node_a>_<node_b>"

    Returns: dict with line names as keys and flow Series as values
    """
    line_flows = {}
    for line in model.dclines():
        try:
            df_line = df_from_pyltm_result(line.transmission_results())

            # Parse line name to get from/to nodes: "dcline_A_B" -> A, B
            parts = line.name.split("_")
            if len(parts) >= 3:
                node_a = parts[1]
                node_b = "_".join(parts[2:])  # Handle node names with underscores
            else:
                continue

            # Check if this line connects to our busbar
            if node_a == busbar_name:
                # Flow out from this busbar (export) should be negative
                flow = -df_line
                other_node = node_b
            elif node_b == busbar_name:
                # Flow in to this busbar (import) should be positive
                flow = df_line
                other_node = node_a
            else:
                continue

            # Extract scenario data
            if scenario == "Mean":
                flow_data = flow.mean(axis=1)
            else:
                flow_data = flow.iloc[:, scenario]

            line_flows[f"{other_node}"] = flow_data
        except Exception:
            # Skip lines that fail to load
            continue

    return line_flows


try:
    # Get individual DC line flows
    dcline_flows_1 = get_connected_dclines(model1, busbar_name, df_load_1.index, scenario_1)
    dcline_flows_2 = get_connected_dclines(model2, busbar_name, df_load_2.index, scenario_2)

    # Calculate net import/export from individual lines
    net_ie_1 = pd.Series(0, index=df_load_1.index)
    for flow_data in dcline_flows_1.values():
        net_ie_1 = net_ie_1 + flow_data

    net_ie_2 = pd.Series(0, index=df_load_2.index)
    for flow_data in dcline_flows_2.values():
        net_ie_2 = net_ie_2 + flow_data

    # Get renewables data using proper API
    renewables_1 = get_renewables_data(model1, busbar_name)
    renewables_2 = get_renewables_data(model2, busbar_name)

    # Check which renewable sources have data
    has_solar = (renewables_1["solar"] is not None and renewables_1["solar"].sum().sum() > 0) or (
        renewables_2["solar"] is not None and renewables_2["solar"].sum().sum() > 0
    )
    has_onshore_wind = (renewables_1["onshore_wind"] is not None and renewables_1["onshore_wind"].sum().sum() > 0) or (
        renewables_2["onshore_wind"] is not None and renewables_2["onshore_wind"].sum().sum() > 0
    )
    has_offshore_wind = (
        renewables_1["offshore_wind"] is not None and renewables_1["offshore_wind"].sum().sum() > 0
    ) or (renewables_2["offshore_wind"] is not None and renewables_2["offshore_wind"].sum().sum() > 0)

    try:
        df_hydro_1 = df_from_pyltm_result(b1.sum_hydro_production())
        df_hydro_2 = df_from_pyltm_result(b2.sum_hydro_production())
        has_hydro = True
    except:
        has_hydro = False

    try:
        df_market_steps_1 = df_from_pyltm_result(b1.sum_production_from_market_steps())
        df_market_steps_2 = df_from_pyltm_result(b2.sum_production_from_market_steps())
        has_market_steps = True
    except:
        has_market_steps = False

    # Extract data based on selected scenarios
    if scenario_1 == "Mean":
        load_1 = df_load_1.mean(axis=1)
        hydro_1 = df_hydro_1.mean(axis=1) if has_hydro else pd.Series(0, index=df_load_1.index)
        solar_1 = (
            renewables_1["solar"].mean(axis=1)
            if has_solar and renewables_1["solar"] is not None
            else pd.Series(0, index=df_load_1.index)
        )
        onshore_wind_1 = (
            renewables_1["onshore_wind"].mean(axis=1)
            if has_onshore_wind and renewables_1["onshore_wind"] is not None
            else pd.Series(0, index=df_load_1.index)
        )
        offshore_wind_1 = (
            renewables_1["offshore_wind"].mean(axis=1)
            if has_offshore_wind and renewables_1["offshore_wind"] is not None
            else pd.Series(0, index=df_load_1.index)
        )
        market_steps_1 = df_market_steps_1.mean(axis=1) if has_market_steps else pd.Series(0, index=df_load_1.index)
    else:
        load_1 = df_load_1.iloc[:, scenario_1]
        hydro_1 = df_hydro_1.iloc[:, scenario_1] if has_hydro else pd.Series(0, index=df_load_1.index)
        solar_1 = (
            renewables_1["solar"].iloc[:, scenario_1]
            if has_solar and renewables_1["solar"] is not None
            else pd.Series(0, index=df_load_1.index)
        )
        onshore_wind_1 = (
            renewables_1["onshore_wind"].iloc[:, scenario_1]
            if has_onshore_wind and renewables_1["onshore_wind"] is not None
            else pd.Series(0, index=df_load_1.index)
        )
        offshore_wind_1 = (
            renewables_1["offshore_wind"].iloc[:, scenario_1]
            if has_offshore_wind and renewables_1["offshore_wind"] is not None
            else pd.Series(0, index=df_load_1.index)
        )
        market_steps_1 = (
            df_market_steps_1.iloc[:, scenario_1] if has_market_steps else pd.Series(0, index=df_load_1.index)
        )

    if scenario_2 == "Mean":
        load_2 = df_load_2.mean(axis=1)
        hydro_2 = df_hydro_2.mean(axis=1) if has_hydro else pd.Series(0, index=df_load_2.index)
        solar_2 = (
            renewables_2["solar"].mean(axis=1)
            if has_solar and renewables_2["solar"] is not None
            else pd.Series(0, index=df_load_2.index)
        )
        onshore_wind_2 = (
            renewables_2["onshore_wind"].mean(axis=1)
            if has_onshore_wind and renewables_2["onshore_wind"] is not None
            else pd.Series(0, index=df_load_2.index)
        )
        offshore_wind_2 = (
            renewables_2["offshore_wind"].mean(axis=1)
            if has_offshore_wind and renewables_2["offshore_wind"] is not None
            else pd.Series(0, index=df_load_2.index)
        )
        market_steps_2 = df_market_steps_2.mean(axis=1) if has_market_steps else pd.Series(0, index=df_load_2.index)
    else:
        load_2 = df_load_2.iloc[:, scenario_2]
        hydro_2 = df_hydro_2.iloc[:, scenario_2] if has_hydro else pd.Series(0, index=df_load_2.index)
        solar_2 = (
            renewables_2["solar"].iloc[:, scenario_2]
            if has_solar and renewables_2["solar"] is not None
            else pd.Series(0, index=df_load_2.index)
        )
        onshore_wind_2 = (
            renewables_2["onshore_wind"].iloc[:, scenario_2]
            if has_onshore_wind and renewables_2["onshore_wind"] is not None
            else pd.Series(0, index=df_load_2.index)
        )
        offshore_wind_2 = (
            renewables_2["offshore_wind"].iloc[:, scenario_2]
            if has_offshore_wind and renewables_2["offshore_wind"] is not None
            else pd.Series(0, index=df_load_2.index)
        )
        market_steps_2 = (
            df_market_steps_2.iloc[:, scenario_2] if has_market_steps else pd.Series(0, index=df_load_2.index)
        )

    # Create stacked area plots for Sim 1 and Sim 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Sim 1 - Scenario {scenario_1}")
        fig_gen_1 = go.Figure()

        # Generation sources (stacked)
        if has_hydro:
            fig_gen_1.add_trace(
                go.Scatter(
                    x=df_load_1.index,
                    y=hydro_1,
                    name="Hydro",
                    stackgroup="generation",
                    fillcolor="rgba(0,100,255,0.6)",
                    line=dict(width=0),
                )
            )

        if has_onshore_wind:
            fig_gen_1.add_trace(
                go.Scatter(
                    x=df_load_1.index,
                    y=onshore_wind_1,
                    name="Onshore Wind",
                    stackgroup="generation",
                    fillcolor="rgba(100,200,100,0.6)",
                    line=dict(width=0),
                )
            )

        if has_offshore_wind:
            fig_gen_1.add_trace(
                go.Scatter(
                    x=df_load_1.index,
                    y=offshore_wind_1,
                    name="Offshore Wind",
                    stackgroup="generation",
                    fillcolor="rgba(50,150,50,0.6)",
                    line=dict(width=0),
                )
            )

        if has_solar:
            fig_gen_1.add_trace(
                go.Scatter(
                    x=df_load_1.index,
                    y=solar_1,
                    name="Solar",
                    stackgroup="generation",
                    fillcolor="rgba(255,200,0,0.6)",
                    line=dict(width=0),
                )
            )

        if has_market_steps:
            fig_gen_1.add_trace(
                go.Scatter(
                    x=df_load_1.index,
                    y=market_steps_1,
                    name="Market Steps",
                    stackgroup="generation",
                    fillcolor="rgba(255,128,0,0.6)",
                    line=dict(width=0),
                )
            )

        # Import/Export - either individual lines or net
        if show_individual_lines:
            # Individual DC lines (stacked)
            import_colors = [
                "rgba(128,0,128,0.6)",
                "rgba(255,0,255,0.6)",
                "rgba(180,0,180,0.6)",
                "rgba(200,50,200,0.6)",
                "rgba(150,0,150,0.6)",
                "rgba(220,100,220,0.6)",
            ]
            for idx, (line_name, flow_data) in enumerate(sorted(dcline_flows_1.items())):
                color = import_colors[idx % len(import_colors)]
                fig_gen_1.add_trace(
                    go.Scatter(
                        x=df_load_1.index,
                        y=flow_data,
                        name=f"Flow: {line_name}",
                        stackgroup="generation",
                        fillcolor=color,
                        line=dict(width=0),
                    )
                )
        else:
            # Net Import/Export (stacked)
            fig_gen_1.add_trace(
                go.Scatter(
                    x=df_load_1.index,
                    y=net_ie_1,
                    name="Net Import (+) / Export (-)",
                    stackgroup="generation",
                    fillcolor="rgba(128,0,128,0.6)",
                    line=dict(width=0),
                )
            )

        # Load (positive, shown as line) - plotted last so it appears on top
        fig_gen_1.add_trace(
            go.Scatter(x=df_load_1.index, y=load_1, name="Load", line=dict(color="red", width=3), mode="lines")
        )

        fig_gen_1.add_hline(y=0, line=dict(color="black", width=1))
        fig_gen_1.update_layout(
            title="Generation Mix vs Load", xaxis_title="Time", yaxis_title="MW", hovermode="x unified", height=500
        )
        st.plotly_chart(fig_gen_1, use_container_width=True)

    with col2:
        st.subheader(f"Sim 2 - Scenario {scenario_2}")
        fig_gen_2 = go.Figure()

        # Generation sources (stacked)
        if has_hydro:
            fig_gen_2.add_trace(
                go.Scatter(
                    x=df_load_2.index,
                    y=hydro_2,
                    name="Hydro",
                    stackgroup="generation",
                    fillcolor="rgba(0,100,255,0.6)",
                    line=dict(width=0),
                )
            )

        if has_onshore_wind:
            fig_gen_2.add_trace(
                go.Scatter(
                    x=df_load_2.index,
                    y=onshore_wind_2,
                    name="Onshore Wind",
                    stackgroup="generation",
                    fillcolor="rgba(100,200,100,0.6)",
                    line=dict(width=0),
                )
            )

        if has_offshore_wind:
            fig_gen_2.add_trace(
                go.Scatter(
                    x=df_load_2.index,
                    y=offshore_wind_2,
                    name="Offshore Wind",
                    stackgroup="generation",
                    fillcolor="rgba(50,150,50,0.6)",
                    line=dict(width=0),
                )
            )

        if has_solar:
            fig_gen_2.add_trace(
                go.Scatter(
                    x=df_load_2.index,
                    y=solar_2,
                    name="Solar",
                    stackgroup="generation",
                    fillcolor="rgba(255,200,0,0.6)",
                    line=dict(width=0),
                )
            )

        if has_market_steps:
            fig_gen_2.add_trace(
                go.Scatter(
                    x=df_load_2.index,
                    y=market_steps_2,
                    name="Market Steps",
                    stackgroup="generation",
                    fillcolor="rgba(255,128,0,0.6)",
                    line=dict(width=0),
                )
            )

        # Import/Export - either individual lines or net
        if show_individual_lines:
            # Individual DC lines (stacked)
            import_colors = [
                "rgba(128,0,128,0.6)",
                "rgba(255,0,255,0.6)",
                "rgba(180,0,180,0.6)",
                "rgba(200,50,200,0.6)",
                "rgba(150,0,150,0.6)",
                "rgba(220,100,220,0.6)",
            ]
            for idx, (line_name, flow_data) in enumerate(sorted(dcline_flows_2.items())):
                color = import_colors[idx % len(import_colors)]
                fig_gen_2.add_trace(
                    go.Scatter(
                        x=df_load_2.index,
                        y=flow_data,
                        name=f"Flow: {line_name}",
                        stackgroup="generation",
                        fillcolor=color,
                        line=dict(width=0),
                    )
                )
        else:
            # Net Import/Export (stacked)
            fig_gen_2.add_trace(
                go.Scatter(
                    x=df_load_2.index,
                    y=net_ie_2,
                    name="Net Import (+) / Export (-)",
                    stackgroup="generation",
                    fillcolor="rgba(128,0,128,0.6)",
                    line=dict(width=0),
                )
            )

        # Load (positive, shown as line) - plotted last so it appears on top
        fig_gen_2.add_trace(
            go.Scatter(x=df_load_2.index, y=load_2, name="Load", line=dict(color="red", width=3), mode="lines")
        )

        fig_gen_2.add_hline(y=0, line=dict(color="black", width=1))
        fig_gen_2.update_layout(
            title="Generation Mix vs Load", xaxis_title="Time", yaxis_title="MW", hovermode="x unified", height=500
        )
        st.plotly_chart(fig_gen_2, use_container_width=True)

    # Summary statistics
    st.subheader("Average Values (MW)")
    metrics = ["Load", "Hydro", "Onshore Wind", "Offshore Wind", "Solar", "Market Steps"]
    sim1_values = [
        float(load_1.mean()),
        float(hydro_1.mean()),
        float(onshore_wind_1.mean()) if has_onshore_wind else 0.0,
        float(offshore_wind_1.mean()) if has_offshore_wind else 0.0,
        float(solar_1.mean()) if has_solar else 0.0,
        float(market_steps_1.mean()) if has_market_steps else 0.0,
    ]
    sim2_values = [
        float(load_2.mean()),
        float(hydro_2.mean()),
        float(onshore_wind_2.mean()) if has_onshore_wind else 0.0,
        float(offshore_wind_2.mean()) if has_offshore_wind else 0.0,
        float(solar_2.mean()) if has_solar else 0.0,
        float(market_steps_2.mean()) if has_market_steps else 0.0,
    ]

    # Add DC line flows based on display mode
    if show_individual_lines:
        # Add individual DC line flows
        all_lines = sorted(set(dcline_flows_1.keys()) | set(dcline_flows_2.keys()))
        total_ie_1 = 0.0
        total_ie_2 = 0.0
        for line_name in all_lines:
            metrics.append(f"Flow: {line_name}")
            flow1 = float(dcline_flows_1[line_name].mean()) if line_name in dcline_flows_1 else 0.0
            flow2 = float(dcline_flows_2[line_name].mean()) if line_name in dcline_flows_2 else 0.0
            sim1_values.append(flow1)
            sim2_values.append(flow2)
            total_ie_1 += flow1
            total_ie_2 += flow2
    else:
        # Add net import/export
        metrics.append("Net Import/Export")
        total_ie_1 = float(net_ie_1.mean())
        total_ie_2 = float(net_ie_2.mean())
        sim1_values.append(total_ie_1)
        sim2_values.append(total_ie_2)

    # Add totals
    metrics.append("Total Generation + Import")
    total_gen_1 = (
        float(hydro_1.mean())
        + (float(onshore_wind_1.mean()) if has_onshore_wind else 0.0)
        + (float(offshore_wind_1.mean()) if has_offshore_wind else 0.0)
        + (float(solar_1.mean()) if has_solar else 0.0)
        + (float(market_steps_1.mean()) if has_market_steps else 0.0)
        + total_ie_1
    )
    total_gen_2 = (
        float(hydro_2.mean())
        + (float(onshore_wind_2.mean()) if has_onshore_wind else 0.0)
        + (float(offshore_wind_2.mean()) if has_offshore_wind else 0.0)
        + (float(solar_2.mean()) if has_solar else 0.0)
        + (float(market_steps_2.mean()) if has_market_steps else 0.0)
        + total_ie_2
    )
    sim1_values.append(total_gen_1)
    sim2_values.append(total_gen_2)

    summary_data = {"Metric": metrics, "Sim 1": sim1_values, "Sim 2": sim2_values}
    df_summary = pd.DataFrame(summary_data)
    df_summary["Difference (Sim2 - Sim1)"] = df_summary["Sim 2"] - df_summary["Sim 1"]
    st.dataframe(
        df_summary.style.format({"Sim 1": "{:.1f}", "Sim 2": "{:.1f}", "Difference (Sim2 - Sim1)": "{:.1f}"}),
        hide_index=True,
    )

except Exception as e:
    st.error(f"Error computing generation mix: {e}")

# --- Market Step Comparison ---
st.markdown("## Market Steps")
df_market_steps_1 = get_market_steps(selected_path=sim1_path, selected_nodes=[busbar_name])
df_market_steps_2 = get_market_steps(selected_path=sim2_path, selected_nodes=[busbar_name])

c1, c2 = st.columns(2)
with c1:
    st.write("Sim1")
    st.dataframe(df_market_steps_1)

with c2:
    st.write("Sim2")
    st.dataframe(df_market_steps_2)

# --- Price Comparison ---
st.header(f"💶 Price Comparison for {busbar_name}")
df_price_1 = df_from_pyltm_result(b1.market_result_price())
df_price_2 = df_from_pyltm_result(b2.market_result_price())

fig_price = go.Figure()

# Sim 1
if use_flattened:
    # Show min, max, mean, median bands
    price_min_1 = df_price_1.min(axis=1)
    price_max_1 = df_price_1.max(axis=1)
    price_mean_1 = df_price_1.mean(axis=1)
    price_median_1 = df_price_1.median(axis=1)

    fig_price.add_trace(
        go.Scatter(
            y=price_max_1, x=df_price_1.index, fill=None, mode="lines", line_color="rgba(0,100,200,0)", showlegend=False
        )
    )
    fig_price.add_trace(
        go.Scatter(
            y=price_min_1,
            x=df_price_1.index,
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,100,200,0)",
            name="Sim 1 (min-max)",
            fillcolor="rgba(0,100,200,0.2)",
        )
    )
    fig_price.add_trace(
        go.Scatter(y=price_mean_1, x=df_price_1.index, name="Sim 1 (mean)", line=dict(color="blue", dash="solid"))
    )
    fig_price.add_trace(
        go.Scatter(y=price_median_1, x=df_price_1.index, name="Sim 1 (median)", line=dict(color="blue", dash="dot"))
    )
else:
    price_mean_1 = df_price_1.mean(axis=1)
    fig_price.add_trace(go.Scatter(y=price_mean_1, x=df_price_1.index, name="Sim 1", line=dict(color="blue")))

# Sim 2
if use_flattened:
    price_min_2 = df_price_2.min(axis=1)
    price_max_2 = df_price_2.max(axis=1)
    price_mean_2 = df_price_2.mean(axis=1)
    price_median_2 = df_price_2.median(axis=1)

    fig_price.add_trace(
        go.Scatter(
            y=price_max_2, x=df_price_2.index, fill=None, mode="lines", line_color="rgba(200,100,0,0)", showlegend=False
        )
    )
    fig_price.add_trace(
        go.Scatter(
            y=price_min_2,
            x=df_price_2.index,
            fill="tonexty",
            mode="lines",
            line_color="rgba(200,100,0,0)",
            name="Sim 2 (min-max)",
            fillcolor="rgba(200,100,0,0.2)",
        )
    )
    fig_price.add_trace(
        go.Scatter(y=price_mean_2, x=df_price_2.index, name="Sim 2 (mean)", line=dict(color="red", dash="solid"))
    )
    fig_price.add_trace(
        go.Scatter(y=price_median_2, x=df_price_2.index, name="Sim 2 (median)", line=dict(color="red", dash="dot"))
    )
else:
    price_mean_2 = df_price_2.mean(axis=1)
    fig_price.add_trace(go.Scatter(y=price_mean_2, x=df_price_2.index, name="Sim 2", line=dict(color="red")))

# Mean values for overall stats
mean_price_1 = df_price_1.mean().mean()
mean_price_2 = df_price_2.mean().mean()

# Horizontal lines
fig_price.add_hline(y=mean_price_1, line=dict(color="blue", dash="dash"), name="Mean Sim 1")
fig_price.add_hline(y=mean_price_2, line=dict(color="red", dash="dash"), name="Mean Sim 2")

# Annotations
fig_price.add_annotation(
    xref="paper", x=1.0, y=mean_price_1, text=f"Mean Sim 1: {mean_price_1:.1f}", showarrow=True, font=dict(color="blue")
)
fig_price.add_annotation(
    xref="paper", x=1.0, y=mean_price_2, text=f"Mean Sim 2: {mean_price_2:.1f}", showarrow=True, font=dict(color="red")
)
title_suffix = " (min/max/mean/median)" if use_flattened else " (averaged)"
fig_price.update_layout(title=f"Average Market Price{title_suffix}", xaxis_title="Time", yaxis_title="EUR/MWh")
st.plotly_chart(fig_price, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.write("Sim 1")
    st.write(f"Average price over scenarios: {df_price_1.mean().mean():.2f} EUR/MWh")
    st.write(f"Average std. dev. over scenarios: {df_price_1.std().mean():.2f}")
    st.write(f"Average Coefficient of Variation (CoV): {(df_price_1.std() / df_price_1.mean()).mean():.2f}")

with col2:
    st.write("Sim 2")
    st.write(f"Average price over scenarios: {df_price_2.mean().mean():.2f} EUR/MWh")
    st.write(f"Average std. dev. over scenarios: {df_price_2.std().mean():.2f}")
    st.write(f"Average Coefficient of Variation (CoV): {(df_price_2.std() / df_price_1.mean()).mean():.2f}")

fig = go.Figure()
fig.add_trace(go.Histogram(x=df_price_1.values.flatten(), name="Sim 1", opacity=0.5))
fig.add_trace(go.Histogram(x=df_price_2.values.flatten(), name="Sim 2", opacity=0.5))
fig.update_layout(barmode="overlay", title="Price Distribution Histogram")
fig.update_traces(opacity=0.6)
st.plotly_chart(fig)

# --- Power Flow ---
st.header("Transmission comparison")


def compute_mean_flows(model):
    dcline = []
    for line in model.dclines():
        _, data = line.transmission_results()
        dcline.append(
            {
                "name": line.name,
                "flow": np.array(data).mean(),
            }
        )
    return pd.DataFrame(dcline).set_index("name")


df_line1 = compute_mean_flows(model1)
df_line2 = compute_mean_flows(model2)

df_line = pd.concat([df_line1.rename(columns={"flow": "Sim1"}), df_line2.rename(columns={"flow": "Sim2"})], axis=1)
df_line["Sim1-Sim2"] = df_line["Sim1"] - df_line["Sim2"]

st.dataframe(df_line.style.background_gradient(cmap="coolwarm"))

# --- Hydro Production ---
st.header(f"🚰 Hydro Production Comparison for {busbar_name}")
try:
    df_hydro_1 = df_from_pyltm_result(b1.sum_hydro_production())
    df_hydro_2 = df_from_pyltm_result(b2.sum_hydro_production())

    fig_hydro = go.Figure()
    if use_flattened:
        hydro_min_1 = df_hydro_1.min(axis=1)
        hydro_max_1 = df_hydro_1.max(axis=1)
        hydro_mean_1 = df_hydro_1.mean(axis=1)
        hydro_median_1 = df_hydro_1.median(axis=1)

        fig_hydro.add_trace(
            go.Scatter(
                y=hydro_max_1,
                x=df_hydro_1.index,
                fill=None,
                mode="lines",
                line_color="rgba(0,100,200,0)",
                showlegend=False,
            )
        )
        fig_hydro.add_trace(
            go.Scatter(
                y=hydro_min_1,
                x=df_hydro_1.index,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,100,200,0)",
                name="Sim 1 (min-max)",
                fillcolor="rgba(0,100,200,0.2)",
            )
        )
        fig_hydro.add_trace(
            go.Scatter(y=hydro_mean_1, x=df_hydro_1.index, name="Sim 1 (mean)", line=dict(color="blue", dash="solid"))
        )
        fig_hydro.add_trace(
            go.Scatter(y=hydro_median_1, x=df_hydro_1.index, name="Sim 1 (median)", line=dict(color="blue", dash="dot"))
        )

        hydro_min_2 = df_hydro_2.min(axis=1)
        hydro_max_2 = df_hydro_2.max(axis=1)
        hydro_mean_2 = df_hydro_2.mean(axis=1)
        hydro_median_2 = df_hydro_2.median(axis=1)

        fig_hydro.add_trace(
            go.Scatter(
                y=hydro_max_2,
                x=df_hydro_2.index,
                fill=None,
                mode="lines",
                line_color="rgba(200,100,0,0)",
                showlegend=False,
            )
        )
        fig_hydro.add_trace(
            go.Scatter(
                y=hydro_min_2,
                x=df_hydro_2.index,
                fill="tonexty",
                mode="lines",
                line_color="rgba(200,100,0,0)",
                name="Sim 2 (min-max)",
                fillcolor="rgba(200,100,0,0.2)",
            )
        )
        fig_hydro.add_trace(
            go.Scatter(y=hydro_mean_2, x=df_hydro_2.index, name="Sim 2 (mean)", line=dict(color="red", dash="solid"))
        )
        fig_hydro.add_trace(
            go.Scatter(y=hydro_median_2, x=df_hydro_2.index, name="Sim 2 (median)", line=dict(color="red", dash="dot"))
        )
    else:
        fig_hydro.add_trace(
            go.Scatter(y=df_hydro_1.mean(axis=1), x=df_hydro_1.index, name="Sim 1", line=dict(color="blue"))
        )
        fig_hydro.add_trace(
            go.Scatter(y=df_hydro_2.mean(axis=1), x=df_hydro_2.index, name="Sim 2", line=dict(color="red"))
        )

    hydro_title_suffix = " (min/max/mean/median)" if use_flattened else " (averaged)"
    fig_hydro.update_layout(title=f"Average Hydro Production{hydro_title_suffix}", xaxis_title="Time", yaxis_title="MW")
    st.plotly_chart(fig_hydro, use_container_width=True)
except RuntimeError:
    st.warning("Hydro production data missing for selected busbar.")

# --- Load ---
st.header(f"🔌 Load Comparison for {busbar_name}")
df_load_1 = df_from_pyltm_result(b1.sum_load())
df_load_2 = df_from_pyltm_result(b2.sum_load())

fig_load = go.Figure()
if use_flattened:
    # Compute per-timestamp statistics across runs/columns
    load_min_1 = df_load_1.min(axis=1)
    load_max_1 = df_load_1.max(axis=1)
    load_mean_1 = df_load_1.mean(axis=1)
    load_median_1 = df_load_1.median(axis=1)

    load_min_2 = df_load_2.min(axis=1)
    load_max_2 = df_load_2.max(axis=1)
    load_mean_2 = df_load_2.mean(axis=1)
    load_median_2 = df_load_2.median(axis=1)

    fig_load.add_trace(
        go.Scatter(
            y=load_max_1, x=df_load_1.index, fill=None, mode="lines", line_color="rgba(0,100,200,0)", showlegend=False
        )
    )
    fig_load.add_trace(
        go.Scatter(
            y=load_min_1,
            x=df_load_1.index,
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,100,200,0)",
            name="Sim 1 (min-max)",
            fillcolor="rgba(0,100,200,0.2)",
        )
    )
    fig_load.add_trace(
        go.Scatter(y=load_mean_1, x=df_load_1.index, name="Sim 1 (mean)", line=dict(color="blue", dash="solid"))
    )
    fig_load.add_trace(
        go.Scatter(y=load_median_1, x=df_load_1.index, name="Sim 1 (median)", line=dict(color="blue", dash="dot"))
    )

    fig_load.add_trace(
        go.Scatter(
            y=load_max_2, x=df_load_2.index, fill=None, mode="lines", line_color="rgba(200,100,0,0)", showlegend=False
        )
    )
    fig_load.add_trace(
        go.Scatter(
            y=load_min_2,
            x=df_load_2.index,
            fill="tonexty",
            mode="lines",
            line_color="rgba(200,100,0,0)",
            name="Sim 2 (min-max)",
            fillcolor="rgba(200,100,0,0.2)",
        )
    )
    fig_load.add_trace(
        go.Scatter(y=load_mean_2, x=df_load_2.index, name="Sim 2 (mean)", line=dict(color="red", dash="solid"))
    )
    fig_load.add_trace(
        go.Scatter(y=load_median_2, x=df_load_2.index, name="Sim 2 (median)", line=dict(color="red", dash="dot"))
    )
else:
    fig_load.add_trace(go.Scatter(y=df_load_1.mean(axis=1), x=df_load_1.index, name="Sim 1", line=dict(color="blue")))
    fig_load.add_trace(go.Scatter(y=df_load_2.mean(axis=1), x=df_load_2.index, name="Sim 2", line=dict(color="red")))
load_title_suffix = " (min/max/mean/median)" if use_flattened else " (averaged)"
fig_load.update_layout(title=f"Total Load{load_title_suffix}", xaxis_title="Time", yaxis_title="MW")
st.plotly_chart(fig_load, use_container_width=True)

# --- Reservoir Level ---
st.header(f"🌊 Reservoir Comparison for {busbar_name}")
try:
    b1_gwh_max_volume = 1.0  # get_gwh_max_volume(busbar=b1)
    b2_gwh_max_volume = 1.0  # get_gwh_max_volume(busbar=b2)

    df_res_1 = df_from_pyltm_result(b1.sum_reservoir()) / b1_gwh_max_volume / 1e3 * 100
    df_res_2 = df_from_pyltm_result(b2.sum_reservoir()) / b2_gwh_max_volume / 1e3 * 100

    fig_res = go.Figure()
    if use_flattened:
        res_min_1 = df_res_1.min(axis=1)
        res_max_1 = df_res_1.max(axis=1)
        res_mean_1 = df_res_1.mean(axis=1)
        res_median_1 = df_res_1.median(axis=1)

        res_min_2 = df_res_2.min(axis=1)
        res_max_2 = df_res_2.max(axis=1)
        res_mean_2 = df_res_2.mean(axis=1)
        res_median_2 = df_res_2.median(axis=1)

        fig_res.add_trace(
            go.Scatter(
                y=res_max_1, x=df_res_1.index, fill=None, mode="lines", line_color="rgba(0,100,200,0)", showlegend=False
            )
        )
        fig_res.add_trace(
            go.Scatter(
                y=res_min_1,
                x=df_res_1.index,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,100,200,0)",
                name="Sim 1 (min-max)",
                fillcolor="rgba(0,100,200,0.2)",
            )
        )
        fig_res.add_trace(
            go.Scatter(y=res_mean_1, x=df_res_1.index, name="Sim 1 (mean)", line=dict(color="blue", dash="solid"))
        )
        fig_res.add_trace(
            go.Scatter(y=res_median_1, x=df_res_1.index, name="Sim 1 (median)", line=dict(color="blue", dash="dot"))
        )

        fig_res.add_trace(
            go.Scatter(
                y=res_max_2, x=df_res_2.index, fill=None, mode="lines", line_color="rgba(200,100,0,0)", showlegend=False
            )
        )
        fig_res.add_trace(
            go.Scatter(
                y=res_min_2,
                x=df_res_2.index,
                fill="tonexty",
                mode="lines",
                line_color="rgba(200,100,0,0)",
                name="Sim 2 (min-max)",
                fillcolor="rgba(200,100,0,0.2)",
            )
        )
        fig_res.add_trace(
            go.Scatter(y=res_mean_2, x=df_res_2.index, name="Sim 2 (mean)", line=dict(color="red", dash="solid"))
        )
        fig_res.add_trace(
            go.Scatter(y=res_median_2, x=df_res_2.index, name="Sim 2 (median)", line=dict(color="red", dash="dot"))
        )
    else:
        fig_res.add_trace(go.Scatter(y=df_res_1.mean(axis=1), x=df_res_1.index, name="Sim 1", line=dict(color="blue")))
        fig_res.add_trace(go.Scatter(y=df_res_2.mean(axis=1), x=df_res_2.index, name="Sim 2", line=dict(color="red")))

    res_title_suffix = " (min/max/mean/median)" if use_flattened else " (averaged)"
    fig_res.update_layout(
        title=f"Mean Reservoir Level{res_title_suffix}", xaxis_title="Time", yaxis_title="Filling [%]"
    )
    st.plotly_chart(fig_res, use_container_width=True)
except AttributeError:
    st.warning("Reservoir data missing for selected busbar.")


# %%
