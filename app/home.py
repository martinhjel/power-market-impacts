from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lpr_sintef_bifrost.ltm import LTM
from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

from app.utils.reservoirs import get_gwh_max_volume

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="EMPS", initial_sidebar_state="expanded")
st.title("🔍 EMPS Model Results Viewer")


# --- FILE SELECTION ---
# st.info(f"Default result path: `{DEFAULT_RESULTS_PATH}`")
# results_path = st.text_input("Path to LTM output folder", DEFAULT_RESULTS_PATH)

result_paths = sorted([i for i in (Path.cwd() / "ltm_output").glob("*/*") if i.is_dir()], key=lambda p: p.name.lower())

# mounted_paths = sorted([i for i in Path("/mnt/empsfileshare/").glob("*/*") if i.is_dir()], key=lambda p: p.name.lower())
# result_paths = mounted_paths


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


path = st.sidebar.selectbox("Select results: ", result_paths, format_func=_result_label)
st.sidebar.markdown(f"Using path: {path}")

if not path.exists():
    st.warning("Please enter a valid path to EMPS results.")
    st.stop()

st.markdown("""
Welcome to the EMPS model input and result visualiser built with **lpr-sintef-bifrost** and **pyltmapi**.
""")

# %% pyltm results


# %%
@st.cache_resource
def read_pyltmapi_model(path):
    return LTM.session_from_folder(path / "run_folder/emps")


st.markdown("## PyLTMAPI results")

with st.spinner("Reading PyLTMAPI model.. ", show_time=True):
    pyltm_session = read_pyltmapi_model(path)
    pyltm_model = pyltm_session.model

    busbars = pyltm_model.busbars()
    busbars_dict = {b.name: b for b in busbars}
    dclines = list(pyltm_model.dclines())
    dclines_dict = {dcline.name: dcline for dcline in dclines}

nodes_csv = Path.cwd() / "app/data/nodes_location.csv"
df_nodes = df_nodes = pd.read_csv(nodes_csv, index_col="id")

# %% Price map
price = {}
for busbar in busbars_dict:
    df_price = df_from_pyltm_result(busbars_dict[busbar].market_result_price())
    df_load = df_from_pyltm_result(busbars_dict[busbar].sum_load())

    weighted_price = ((df_price * df_load).sum(axis=1) / df_load.sum(axis=1)).mean()
    price[busbar] = weighted_price


# Convert to DataFrame for plotting
df_price_map = pd.DataFrame.from_dict(price, orient="index", columns=["avg_weighted_price"])
df_price_map.index.name = "id"

# Merge with node coordinates
df_nodes_plot = df_nodes.join(df_price_map, how="inner")

# Create Plotly map
fig = go.Figure()
fig.add_trace(
    go.Scattergeo(
        lon=df_nodes_plot["lon"],
        lat=df_nodes_plot["lat"],
        mode="markers+text",
        text=[
            f"<b>{node}<br>{price:.1f} €/MWh<b>"
            for node, price in zip(df_nodes_plot.index, df_nodes_plot["avg_weighted_price"])
        ],
        textposition="top center",
        marker=dict(
            size=12,
            color=df_nodes_plot["avg_weighted_price"],
            colorscale="RdBu_r",
            colorbar=dict(title="Avg Weighted Price"),
            line=dict(color="black", width=0.5),
        ),
        hoverinfo="text",
        showlegend=False,
    )
)

fig.update_geos(
    projection_type="mercator",
    showcountries=True,
    showland=True,
    landcolor="rgb(240,240,240)",
    lataxis_range=[50, 72],
    lonaxis_range=[-5, 35],
)
fig.update_layout(
    margin=dict(l=0, r=0, t=40, b=0),
    title_text="Average Weighted Electricity Price per Node",
    width=1200,
    height=800,
)

st.plotly_chart(fig)

# dcline = dclines_dict[dcline_name]

# df_line = df_from_pyltm_result(dcline.transmission_results())

# %% DC line utilization map
st.markdown("### ⚡ DC Line Utilization & Direction")
dc_records = []
dc_metrics = []
for line in dclines:
    if line.from_busbar not in df_nodes.index or line.to_busbar not in df_nodes.index:
        continue

    df_flow = df_from_pyltm_result(line.transmission_results())
    flow_avg = df_flow.mean(axis=1)
    avg_flow = float(flow_avg.mean())
    avg_abs_flow = float(np.abs(flow_avg).mean())
    max_abs_flow = float(np.abs(flow_avg).max())
    net_energy_mwh = float(flow_avg.sum())

    forward_capacity = (
        float(line.forward_capacity.scenarios[0][0])
        if getattr(line, "forward_capacity", None) and line.forward_capacity.scenarios
        else 0.0
    )
    backward_capacity = (
        float(line.backward_capacity.scenarios[0][0])
        if getattr(line, "backward_capacity", None) and line.backward_capacity.scenarios
        else 0.0
    )
    max_capacity = max(forward_capacity, backward_capacity)
    utilization_pct = (max_abs_flow / max_capacity * 100.0) if max_capacity else 0.0

    lon_from, lat_from = df_nodes.loc[line.from_busbar, ["lon", "lat"]]
    lon_to, lat_to = df_nodes.loc[line.to_busbar, ["lon", "lat"]]

    flow_direction = (
        f"{line.from_busbar} → {line.to_busbar}" if avg_flow >= 0 else f"{line.to_busbar} → {line.from_busbar}"
    )

    dc_records.append(
        {
            "Line": line.name,
            "From": line.from_busbar,
            "To": line.to_busbar,
            "Avg flow (MW)": avg_flow,
            "Avg |flow| (MW)": avg_abs_flow,
            "Peak |flow| (MW)": max_abs_flow,
            "Capacity (MW)": max_capacity,
            "Utilisation (%)": utilization_pct,
            "Direction": flow_direction,
            "Net energy (GWh)": net_energy_mwh / 1000.0,
        }
    )

    dc_metrics.append(
        {
            "line": line,
            "lon_from": lon_from,
            "lat_from": lat_from,
            "lon_to": lon_to,
            "lat_to": lat_to,
            "avg_flow": avg_flow,
            "avg_abs_flow": avg_abs_flow,
            "max_abs_flow": max_abs_flow,
            "max_capacity": max_capacity,
            "utilization_pct": utilization_pct,
            "net_energy_mwh": net_energy_mwh,
            "flow_direction": flow_direction,
        }
    )

dc_line_traces = []
arrow_traces = []

if dc_metrics:
    max_util = max(metric["utilization_pct"] for metric in dc_metrics) or 1.0
    max_capacity_overall = max(metric["max_capacity"] for metric in dc_metrics) or 1.0
    util_colorscale = "Viridis"

    for idx_metric, metric in enumerate(dc_metrics):
        lon_from = metric["lon_from"]
        lat_from = metric["lat_from"]
        lon_to = metric["lon_to"]
        lat_to = metric["lat_to"]
        avg_flow = metric["avg_flow"]
        util_pct = metric["utilization_pct"]
        max_capacity = metric["max_capacity"]
        net_energy_mwh = metric["net_energy_mwh"]
        avg_abs_flow = metric["avg_abs_flow"]
        max_abs_flow = metric["max_abs_flow"]
        flow_direction = metric["flow_direction"]

        capacity_ratio = metric["max_capacity"] / max_capacity_overall if max_capacity_overall else 0.0
        color = pc.sample_colorscale(util_colorscale, (util_pct / max_util) if max_util > 0 else 0.0)[0]
        width = 1.5 + 8 * capacity_ratio

        hover_text = (
            f"<b>{metric['line'].name}</b><br>"
            f"Direction: {flow_direction}<br>"
            f"Avg flow: {avg_flow:.1f} MW<br>"
            f"Avg |flow|: {avg_abs_flow:.1f} MW<br>"
            f"Peak |flow|: {max_abs_flow:.1f} MW<br>"
            f"Capacity: {max_capacity:.1f} MW<br>"
            f"Utilisation: {util_pct:.1f}%<br>"
            f"Net energy: {net_energy_mwh / 1000:.1f} GWh"
        )

        dc_line_traces.append(
            go.Scattergeo(
                lon=[lon_from, lon_to],
                lat=[lat_from, lat_to],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="text",
                text=hover_text,
                showlegend=False,
            )
        )

        # Arrow placement and orientation
        if avg_flow >= 0:
            send_lon, send_lat = lon_from, lat_from
            recv_lon, recv_lat = lon_to, lat_to
        else:
            send_lon, send_lat = lon_to, lat_to
            recv_lon, recv_lat = lon_from, lat_from

        arrow_factor = 0.85
        arrow_lon = send_lon + (recv_lon - send_lon) * arrow_factor
        arrow_lat = send_lat + (recv_lat - send_lat) * arrow_factor
        dx = recv_lon - send_lon
        dy = recv_lat - send_lat

        mean_lat = (lat_from + lat_to) / 2
        angle = np.degrees(np.arctan2(dy, dx * np.cos(np.radians(mean_lat)))) if (dx or dy) else 0.0

        marker_kwargs = dict(
            symbol="triangle-up",
            size=12,
            color=util_pct,
            colorscale=util_colorscale,
            cmin=0,
            cmax=max_util,
            showscale=idx_metric == 0,
            line=dict(color="black", width=0.5),
            angle=angle,
            angleref="north",
        )
        if idx_metric == 0:
            marker_kwargs["colorbar"] = dict(title="Utilisation (%)")

        arrow_traces.append(
            go.Scattergeo(
                lon=[arrow_lon],
                lat=[arrow_lat],
                mode="markers",
                marker=marker_kwargs,
                hoverinfo="text",
                text=hover_text,
                showlegend=False,
            )
        )

if dc_line_traces:
    fig_dc = go.Figure(dc_line_traces + arrow_traces)
    fig_dc.add_trace(
        go.Scattergeo(
            lon=df_nodes["lon"],
            lat=df_nodes["lat"],
            mode="markers+text",
            marker=dict(size=6, color="black"),
            text=df_nodes.index,
            textposition="top center",
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig_dc.update_geos(
        projection_type="mercator",
        showcountries=True,
        showland=True,
        landcolor="rgb(240,240,240)",
        lataxis_range=[50, 72],
        lonaxis_range=[-5, 35],
    )
    fig_dc.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title_text="DC Line Average Flow (colour = net direction, width = utilisation)",
        width=1200,
        height=600,
    )
    st.plotly_chart(fig_dc, use_container_width=True)

if dc_records:
    df_dc = pd.DataFrame(dc_records).sort_values("Utilisation (%)", ascending=False)
    st.dataframe(
        df_dc.style.format(
            {
                "Avg flow (MW)": "{:.1f}",
                "Avg |flow| (MW)": "{:.1f}",
                "Peak |flow| (MW)": "{:.1f}",
                "Capacity (MW)": "{:.1f}",
                "Utilisation (%)": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

# %%  Select busbar

busbar_name = st.selectbox("Pick busbar:  ", busbars_dict.keys())
busbar = busbars_dict[busbar_name]

# %%

# --- Generation Mix and Net Import/Export ---
st.subheader("⚡ Generation Mix and Net Import/Export")

# Scenario selection and display options
col1, col2 = st.columns(2)
with col1:
    df_load = df_from_pyltm_result(busbar.sum_load())
    scenario = st.selectbox("Select scenario:", ["Mean"] + list(range(df_load.shape[1])), key="gen_mix_scenario")
with col2:
    show_individual_lines = st.checkbox("Show individual DC lines", value=False)


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
    dcline_flows = get_connected_dclines(pyltm_model, busbar_name, df_load.index, scenario)

    # Calculate net import/export from individual lines
    net_ie = pd.Series(0, index=df_load.index)
    for flow_data in dcline_flows.values():
        net_ie = net_ie + flow_data

    # Get renewables data using proper API
    renewables = get_renewables_data(pyltm_model, busbar_name)

    # Check which renewable sources have data
    has_solar = renewables["solar"] is not None and renewables["solar"].sum().sum() > 0
    has_onshore_wind = renewables["onshore_wind"] is not None and renewables["onshore_wind"].sum().sum() > 0
    has_offshore_wind = renewables["offshore_wind"] is not None and renewables["offshore_wind"].sum().sum() > 0

    # Get hydro and market steps data
    try:
        df_hydro = df_from_pyltm_result(busbar.sum_hydro_production())
        has_hydro = True
    except:
        has_hydro = False

    try:
        df_market_steps = df_from_pyltm_result(busbar.sum_production_from_market_steps())
        has_market_steps = True
    except:
        has_market_steps = False

    # Extract data based on selected scenario
    if scenario == "Mean":
        load_data = df_load.mean(axis=1)
        hydro_data = df_hydro.mean(axis=1) if has_hydro else pd.Series(0, index=df_load.index)
        solar_data = renewables["solar"].mean(axis=1) if has_solar else pd.Series(0, index=df_load.index)
        onshore_wind_data = (
            renewables["onshore_wind"].mean(axis=1) if has_onshore_wind else pd.Series(0, index=df_load.index)
        )
        offshore_wind_data = (
            renewables["offshore_wind"].mean(axis=1) if has_offshore_wind else pd.Series(0, index=df_load.index)
        )
        market_steps_data = df_market_steps.mean(axis=1) if has_market_steps else pd.Series(0, index=df_load.index)
    else:
        load_data = df_load.iloc[:, scenario]
        hydro_data = df_hydro.iloc[:, scenario] if has_hydro else pd.Series(0, index=df_load.index)
        solar_data = renewables["solar"].iloc[:, scenario] if has_solar else pd.Series(0, index=df_load.index)
        onshore_wind_data = (
            renewables["onshore_wind"].iloc[:, scenario] if has_onshore_wind else pd.Series(0, index=df_load.index)
        )
        offshore_wind_data = (
            renewables["offshore_wind"].iloc[:, scenario] if has_offshore_wind else pd.Series(0, index=df_load.index)
        )
        market_steps_data = df_market_steps.iloc[:, scenario] if has_market_steps else pd.Series(0, index=df_load.index)

    # Create stacked area plot
    fig_gen = go.Figure()

    # Generation sources (stacked)
    if has_hydro:
        fig_gen.add_trace(
            go.Scatter(
                x=df_load.index,
                y=hydro_data,
                name="Hydro",
                stackgroup="generation",
                fillcolor="rgba(0,100,255,0.6)",
                line=dict(width=0),
            )
        )

    if has_onshore_wind:
        fig_gen.add_trace(
            go.Scatter(
                x=df_load.index,
                y=onshore_wind_data,
                name="Onshore Wind",
                stackgroup="generation",
                fillcolor="rgba(100,200,100,0.6)",
                line=dict(width=0),
            )
        )

    if has_offshore_wind:
        fig_gen.add_trace(
            go.Scatter(
                x=df_load.index,
                y=offshore_wind_data,
                name="Offshore Wind",
                stackgroup="generation",
                fillcolor="rgba(50,150,50,0.6)",
                line=dict(width=0),
            )
        )

    if has_solar:
        fig_gen.add_trace(
            go.Scatter(
                x=df_load.index,
                y=solar_data,
                name="Solar",
                stackgroup="generation",
                fillcolor="rgba(255,200,0,0.6)",
                line=dict(width=0),
            )
        )

    if has_market_steps:
        fig_gen.add_trace(
            go.Scatter(
                x=df_load.index,
                y=market_steps_data,
                name="Market Steps",
                stackgroup="generation",
                fillcolor="rgba(255,128,0,0.6)",
                line=dict(width=0),
            )
        )

    # Import/Export - either individual lines or net
    if show_individual_lines:
        # Individual DC lines (stacked)
        # Color palette for different lines
        import_colors = [
            "rgba(128,0,128,0.6)",
            "rgba(255,0,255,0.6)",
            "rgba(180,0,180,0.6)",
            "rgba(200,50,200,0.6)",
            "rgba(150,0,150,0.6)",
            "rgba(220,100,220,0.6)",
        ]
        for idx, (line_name, flow_data) in enumerate(sorted(dcline_flows.items())):
            color = import_colors[idx % len(import_colors)]
            fig_gen.add_trace(
                go.Scatter(
                    x=df_load.index,
                    y=flow_data,
                    name=f"Flow: {line_name}",
                    stackgroup="generation",
                    fillcolor=color,
                    line=dict(width=0),
                )
            )
    else:
        # Net Import/Export (stacked)
        fig_gen.add_trace(
            go.Scatter(
                x=df_load.index,
                y=net_ie,
                name="Net Import (+) / Export (-)",
                stackgroup="generation",
                fillcolor="rgba(128,0,128,0.6)",
                line=dict(width=0),
            )
        )

    # Load (positive, shown as line) - plotted last so it appears on top
    fig_gen.add_trace(
        go.Scatter(x=df_load.index, y=load_data, name="Load", line=dict(color="red", width=3), mode="lines")
    )

    fig_gen.add_hline(y=0, line=dict(color="black", width=1))
    fig_gen.update_layout(
        title=f"Generation Mix vs Load - Scenario {scenario}",
        xaxis_title="Time",
        yaxis_title="MW",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig_gen, use_container_width=True)

    # Summary statistics
    st.subheader("Average Values (MW)")
    metrics = ["Load", "Hydro", "Onshore Wind", "Offshore Wind", "Solar", "Market Steps"]
    values = [
        float(load_data.mean()),
        float(hydro_data.mean()) if has_hydro else 0.0,
        float(onshore_wind_data.mean()) if has_onshore_wind else 0.0,
        float(offshore_wind_data.mean()) if has_offshore_wind else 0.0,
        float(solar_data.mean()) if has_solar else 0.0,
        float(market_steps_data.mean()) if has_market_steps else 0.0,
    ]

    # Add DC line flows based on display mode
    if show_individual_lines:
        # Add individual DC line flows
        for line_name, flow_data in sorted(dcline_flows.items()):
            avg_flow = float(flow_data.mean())
            metrics.append(f"Flow: {line_name}")
            values.append(avg_flow)
    else:
        # Add net import/export
        metrics.append("Net Import/Export")
        values.append(float(net_ie.mean()))

    # Add totals
    metrics.append("Total Generation + Import")
    total_gen = (
        (float(hydro_data.mean()) if has_hydro else 0.0)
        + (float(onshore_wind_data.mean()) if has_onshore_wind else 0.0)
        + (float(offshore_wind_data.mean()) if has_offshore_wind else 0.0)
        + (float(solar_data.mean()) if has_solar else 0.0)
        + (float(market_steps_data.mean()) if has_market_steps else 0.0)
        + float(net_ie.mean())
    )
    values.append(total_gen)

    summary_data = {"Metric": metrics, "Value": values}
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary.style.format({"Value": "{:.1f}"}), hide_index=True)

except Exception as e:
    st.error(f"Error computing generation mix: {e}")

# %% DC lines results
st.markdown("### DC Lines Results")

if st.checkbox("Visualize dc lines results ", value=False):
    dcline_name = st.selectbox("Choose line: ", dclines_dict.keys())
    # dcline_name = "dcline_DK1 NO2"
    dcline = dclines_dict[dcline_name]

    df_line = df_from_pyltm_result(dcline.transmission_results())
    fig = px.line(df_line, title=f"DC Line {dcline_name}", labels={"index": "Time", "value": "Flow"})
    st.plotly_chart(fig, use_container_width=True)

# %% Busbar


def plot_water_value(busbar):
    try:
        wv_results = busbar.water_value()
    except Exception as exc:
        raise RuntimeError(f"Water value results unavailable for {busbar.name}") from exc

    if not wv_results or len(wv_results) != 2:
        raise RuntimeError(f"Unexpected water value payload for {busbar.name}")

    time, values = wv_results

    time_np = np.array(time, dtype="datetime64[ms]")
    water_values = np.array(values, copy=False)

    if water_values.shape != ():
        if water_values.ndim < 4:
            raise RuntimeError(f"Water value array has unsupported shape {water_values.shape}")
        wv_for_plotting = water_values[:, 0, 0, :]
        max_reservoir_level = 100
        filling = np.linspace(max_reservoir_level, 0, wv_for_plotting.shape[1])

        X, Y = np.meshgrid(time_np, filling)

        # Compute log10 of z for coloring, but keep original for hover
        z = wv_for_plotting.T
        log_z = np.log10(np.clip(z, 1e-2, None))  # avoid log(0)

        fig = go.Figure(
            data=go.Contour(
                z=log_z,
                x=time_np,
                y=filling,
                colorscale="Viridis",
                colorbar=dict(
                    title="Water Value",
                    tickvals=np.log10([0.01, 0.1, 1, 10, 100, 1000]),
                    ticktext=["0.01", "0.1", "1", "10", "100", "1000"],
                ),
                contours=dict(coloring="heatmap", showlabels=False),
                hovertemplate="Filling: %{y:.1f}%<br>Time: %{x|%Y-%m-%d}<br>Value: %{customdata:.2f}",
                customdata=z,
            )
        )
        fig.update_layout(
            title=f"Water Values (log color scale): {busbar.name}",
            yaxis_title="Filling (%)",
            xaxis_title="Week",
        )

        return fig


st.markdown("### Busbar Results")

if st.checkbox("Visualize busbar results", value=False):
    # --- Other busbar results ---
    # [i for i in dir(busbar) if "__" not in i]
    if busbar.have_detailed_hydro_results:
        try:
            fig = plot_water_value(busbar)
            st.plotly_chart(fig, use_container_width=True)
            gwh_max_volume = get_gwh_max_volume(busbar=busbar)

            df_reservoir = df_from_pyltm_result(busbar.reservoir()) / gwh_max_volume / 1e3 * 100
            df_reservoir["MEAN"] = df_reservoir.mean(axis=1)
            fig = px.line(
                df_reservoir, title=f"Reservoir {busbar.name}", labels={"index": "Time", "value": "Reservoir"}
            )
            fig.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        except RuntimeError:
            st.error(f"Cannot plot water values or aggregated reservoir for {busbar.name}.")

    df_price = df_from_pyltm_result(busbar.market_result_price())
    df_price["MEAN"] = df_price.mean(axis=1)
    fig = px.line(
        df_price, title=f"Market Result Price {busbar.name}", labels={"index": "Time", "value": "Market Price"}
    )
    st.plotly_chart(fig, use_container_width=True)

    try:
        df_hydro = df_from_pyltm_result(busbar.sum_hydro_production())
        df_hydro["MEAN"] = df_hydro.mean(axis=1)
        fig = px.line(
            df_hydro, title=f"Sum Hydro Production {busbar.name}", labels={"index": "Time", "value": "Production"}
        )
        st.plotly_chart(fig, use_container_width=True)
    except RuntimeError:
        st.error(f"Cannot plot hydro production for {busbar.name}.")

    df_load = df_from_pyltm_result(busbar.sum_load())
    fig = px.line(df_load, title=f"Sum Load {busbar.name}", labels={"index": "Time", "value": "Load"})
    st.plotly_chart(fig, use_container_width=True)

    df_ms = df_from_pyltm_result(busbar.sum_production_from_market_steps())
    fig = px.line(
        df_ms, title=f"Sum Production from Market Steps {busbar.name}", labels={"index": "Time", "value": "Production"}
    )
    st.plotly_chart(fig, use_container_width=True)


# %% Aggregated hydro resukts
st.markdown("### Aggregated Hydro Results")

if st.checkbox("Visualize aggregated reservoir", value=False):
    if not busbar.have_aggregated_hydro_results():
        st.error("The selected busbar does not have aggregated hydro reservoirs.")
        st.stop()

    agg_hydros = pyltm_model.aggregated_hydro_modules()
    agg_hydros_dict = {agg_hydro.name: agg_hydro for agg_hydro in agg_hydros}

    agg_hydro_name = st.selectbox("Select aggregated hydro: ", agg_hydros_dict.keys())
    agg_hydro = agg_hydros_dict[agg_hydro_name]
    # agg_hydro = agg_hydros[1]

    data = np.array(agg_hydro.regulated_energy_inflow.scenarios)
    index = agg_hydro.regulated_energy_inflow.timestamps
    unit = agg_hydro.regulated_energy_inflow.unit

    df = pd.DataFrame(data.T, index=index)

    fig = px.line(df, title=f"Regulated energy inflow {agg_hydro.name}", labels={"index": "Time", "value": f"{unit}"})
    st.plotly_chart(fig, use_container_width=True)

    st.write("Start Reservoir Energy, Unit: MWh")
    st.write(agg_hydro.start_reservoir_energy)

    st.write("Station Power, Unit : MW")
    st.write(agg_hydro.station_power)

    st.write("Reservoir Energy")
    st.write(agg_hydro.reservoir_energy)

# %% Detailed hydro results

st.markdown("### Detailed Hydro Results")

if st.checkbox("Visualize detailed reservoir", value=False):
    if not busbar.have_detailed_hydro_results():
        st.error("The selected busbar does not have detailed hydro reservoirs.")
        st.stop()
    reservoirs_dict = {r.name: r for r in busbar.reservoirs()}
    reservoir_name = st.selectbox("Select reservoir: ", reservoirs_dict.keys())

    rsv = reservoirs_dict[reservoir_name]
    df_res = df_from_pyltm_result(rsv.reservoir(time_axis=True))
    df_prod = df_from_pyltm_result(rsv.production(time_axis=True))
    df_dis = df_from_pyltm_result(rsv.discharge(time_axis=True))
    df_spill = df_from_pyltm_result(rsv.spill(time_axis=True))
    df_inflow = df_from_pyltm_result(rsv.inflow(time_axis=True))
    df_bypass = df_from_pyltm_result(rsv.bypass(time_axis=True))

    # --- Reservoir level ---
    fig_res = px.line(df_res, title=f"Reservoir {rsv.name}", labels={"index": "Time", "value": "Reservoir level"})
    st.plotly_chart(fig_res, use_container_width=True)

    # --- Production ---
    fig_prod = px.line(df_prod, title=f"Production {rsv.name}", labels={"index": "Time", "value": "Production"})
    st.plotly_chart(fig_prod, use_container_width=True)

    # --- Discharge ---
    fig_dis = px.line(df_dis, title=f"Discharge {rsv.name}", labels={"index": "Time", "value": "Discharge"})
    st.plotly_chart(fig_dis, use_container_width=True)

    # --- Spillage ---
    fig_spill = px.line(df_spill, title=f"Spillage: {rsv.name}", labels={"index": "Time", "value": "Spillage"})
    st.plotly_chart(fig_spill, use_container_width=True)

    # --- Inflow ---
    fig_inflow = px.line(df_inflow, title=f"Inflow: {rsv.name}", labels={"index": "Time", "value": "Inflow"})
    st.plotly_chart(fig_inflow, use_container_width=True)

    # --- Bypass ---
    fig_bypass = px.line(df_bypass, title=f"Bypass: {rsv.name}", labels={"index": "Time", "value": "Bypass"})
    st.plotly_chart(fig_bypass, use_container_width=True)
