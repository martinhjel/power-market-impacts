from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.processed import (
    busbar_metric,
    busbar_names,
    connected_dcline_flows,
    dcline_flow,
    dcline_names,
    default_index,
    list_processed_scenarios,
    load_processed_scenario,
    net_import_export,
    parse_dcline_endpoints,
    result_label,
    safe_busbar_metric,
    select_timeseries,
)

st.set_page_config(layout="wide", page_title="EMPS", initial_sidebar_state="expanded")
st.title("EMPS Processed Results Viewer")


paths = list_processed_scenarios()
if not paths:
    st.error("No processed results found under ltm_processed/*/*/processed_data.parquet.")
    st.stop()

selected_path = st.sidebar.selectbox(
    "Select results:",
    paths,
    format_func=result_label,
    index=0,
)
path = str(selected_path)
load_processed_scenario(path)
st.sidebar.markdown(f"Using processed path: `{selected_path}`")

if st.sidebar.button("Reload processed data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

nodes_csv = Path.cwd() / "app/data/nodes_location.csv"
df_nodes = pd.read_csv(nodes_csv, index_col="id")


def metric_or_zero(area: str, metric: str, like: pd.DataFrame) -> pd.DataFrame:
    return safe_busbar_metric(path, area, metric, like=like)


def has_energy(df: pd.DataFrame) -> bool:
    return bool(df.abs().sum().sum() > 1e-6)


@st.cache_data(show_spinner="Computing weighted prices...")
def compute_avg_weighted_prices(processed_path: str) -> dict[str, float]:
    prices: dict[str, float] = {}
    for area in busbar_names(processed_path):
        try:
            price = busbar_metric(processed_path, area, "price")
            load = busbar_metric(processed_path, area, "load")
        except KeyError:
            continue
        load_sum = load.sum(axis=1)
        valid = load_sum != 0
        if valid.any():
            weighted = price.mul(load, fill_value=0.0).sum(axis=1)[valid] / load_sum[valid]
            prices[area] = float(weighted.mean())
        else:
            prices[area] = float("nan")
    return prices


st.markdown("## Price Map")
price_map = compute_avg_weighted_prices(path)
df_price_map = pd.DataFrame.from_dict(price_map, orient="index", columns=["avg_weighted_price"])
df_price_map.index.name = "id"
df_nodes_plot = df_nodes.join(df_price_map, how="inner")

if df_nodes_plot.empty:
    st.info("No mapped nodes found for the selected processed scenario.")
else:
    fig_price_map = go.Figure()
    fig_price_map.add_trace(
        go.Scattergeo(
            lon=df_nodes_plot["lon"],
            lat=df_nodes_plot["lat"],
            mode="markers+text",
            text=[
                f"<b>{node}</b><br>{price:.1f} EUR/MWh"
                for node, price in zip(df_nodes_plot.index, df_nodes_plot["avg_weighted_price"])
            ],
            textposition="top center",
            marker=dict(
                size=12,
                color=df_nodes_plot["avg_weighted_price"],
                colorscale="RdBu_r",
                colorbar=dict(title="Avg price"),
                line=dict(color="black", width=0.5),
            ),
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig_price_map.update_geos(
        projection_type="mercator",
        showcountries=True,
        showland=True,
        landcolor="rgb(240,240,240)",
        lataxis_range=[50, 72],
        lonaxis_range=[-5, 35],
    )
    fig_price_map.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title_text="Average Weighted Electricity Price per Node",
        height=700,
    )
    st.plotly_chart(fig_price_map, width="stretch")


st.markdown("## DC Line Flow Map")
dc_records = []
dc_metrics = []
for line_name in dcline_names(path):
    endpoints = parse_dcline_endpoints(line_name)
    if endpoints is None:
        continue
    node_a, node_b = endpoints
    if node_a not in df_nodes.index or node_b not in df_nodes.index:
        continue
    try:
        flow = dcline_flow(path, line_name)
    except KeyError:
        continue

    flow_avg = flow.mean(axis=1)
    avg_flow = float(flow_avg.mean())
    avg_abs_flow = float(flow_avg.abs().mean())
    max_abs_flow = float(flow_avg.abs().max())
    net_energy_gwh = float(flow_avg.sum() / 1000.0)
    flow_direction = f"{node_a} -> {node_b}" if avg_flow >= 0 else f"{node_b} -> {node_a}"
    lon_from, lat_from = df_nodes.loc[node_a, ["lon", "lat"]]
    lon_to, lat_to = df_nodes.loc[node_b, ["lon", "lat"]]

    dc_records.append(
        {
            "Line": line_name,
            "From": node_a,
            "To": node_b,
            "Avg flow (MW)": avg_flow,
            "Avg |flow| (MW)": avg_abs_flow,
            "Peak |flow| (MW)": max_abs_flow,
            "Direction": flow_direction,
            "Net energy (GWh)": net_energy_gwh,
        }
    )
    dc_metrics.append(
        {
            "line_name": line_name,
            "node_a": node_a,
            "node_b": node_b,
            "lon_from": lon_from,
            "lat_from": lat_from,
            "lon_to": lon_to,
            "lat_to": lat_to,
            "avg_flow": avg_flow,
            "avg_abs_flow": avg_abs_flow,
            "max_abs_flow": max_abs_flow,
            "flow_direction": flow_direction,
            "net_energy_gwh": net_energy_gwh,
        }
    )

if dc_metrics:
    max_abs_flow_overall = max(metric["max_abs_flow"] for metric in dc_metrics) or 1.0
    colorscale = "Viridis"
    traces = []
    arrow_traces = []
    for idx, metric in enumerate(dc_metrics):
        flow_ratio = metric["max_abs_flow"] / max_abs_flow_overall if max_abs_flow_overall else 0.0
        color = pc.sample_colorscale(colorscale, flow_ratio)[0]
        width = 1.5 + 8 * flow_ratio
        lon_from = metric["lon_from"]
        lat_from = metric["lat_from"]
        lon_to = metric["lon_to"]
        lat_to = metric["lat_to"]
        hover_text = (
            f"<b>{metric['line_name']}</b><br>"
            f"Direction: {metric['flow_direction']}<br>"
            f"Avg flow: {metric['avg_flow']:.1f} MW<br>"
            f"Avg |flow|: {metric['avg_abs_flow']:.1f} MW<br>"
            f"Peak |flow|: {metric['max_abs_flow']:.1f} MW<br>"
            f"Net energy: {metric['net_energy_gwh']:.1f} GWh"
        )
        traces.append(
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

        if metric["avg_flow"] >= 0:
            send_lon, send_lat = lon_from, lat_from
            recv_lon, recv_lat = lon_to, lat_to
        else:
            send_lon, send_lat = lon_to, lat_to
            recv_lon, recv_lat = lon_from, lat_from
        arrow_lon = send_lon + (recv_lon - send_lon) * 0.85
        arrow_lat = send_lat + (recv_lat - send_lat) * 0.85
        dx = recv_lon - send_lon
        dy = recv_lat - send_lat
        mean_lat = (lat_from + lat_to) / 2
        angle = np.degrees(np.arctan2(dy, dx * np.cos(np.radians(mean_lat)))) if (dx or dy) else 0.0

        marker_kwargs = dict(
            symbol="triangle-up",
            size=12,
            color=metric["max_abs_flow"],
            colorscale=colorscale,
            cmin=0,
            cmax=max_abs_flow_overall,
            showscale=idx == 0,
            line=dict(color="black", width=0.5),
            angle=angle,
            angleref="north",
        )
        if idx == 0:
            marker_kwargs["colorbar"] = dict(title="Peak |flow| MW")
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

    fig_dc = go.Figure(traces + arrow_traces)
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
        title_text="DC Line Average Flow Direction and Peak Flow",
        height=650,
    )
    st.plotly_chart(fig_dc, width="stretch")

if dc_records:
    df_dc = pd.DataFrame(dc_records).sort_values("Peak |flow| (MW)", ascending=False)
    st.dataframe(
        df_dc.style.format(
            {
                "Avg flow (MW)": "{:.1f}",
                "Avg |flow| (MW)": "{:.1f}",
                "Peak |flow| (MW)": "{:.1f}",
                "Net energy (GWh)": "{:.1f}",
            }
        ),
        width="stretch",
    )


areas = busbar_names(path)
default_area = "NO2" if "NO2" in areas else areas[0]
area = st.selectbox("Pick busbar:", areas, index=default_index(areas, default_area))
load = busbar_metric(path, area, "load")
weather_years = ["Mean", *list(load.columns)]

st.markdown("## Generation Mix and Net Import/Export")
col1, col2 = st.columns(2)
with col1:
    weather_year = st.selectbox("Weather year:", weather_years, key="gen_mix_weather_year")
with col2:
    show_individual_lines = st.checkbox("Show individual DC lines", value=False)


def selected(metric: str) -> pd.Series:
    return select_timeseries(metric_or_zero(area, metric, load), weather_year)


fixed_nuclear = selected("fixed_nuclear")
total_nuclear = selected("total_nuclear")
flexible_nuclear_in_market_steps = total_nuclear.sub(fixed_nuclear, fill_value=0.0).clip(lower=0.0)
other_market_steps = selected("market_steps").sub(flexible_nuclear_in_market_steps, fill_value=0.0)

component_specs = [
    ("hydro", "Hydro", "rgba(0,100,255,0.6)"),
    ("onshore_wind", "Onshore Wind", "rgba(100,190,100,0.6)"),
    ("offshore_wind", "Offshore Wind", "rgba(50,145,50,0.6)"),
    ("solar", "Solar", "rgba(245,190,0,0.6)"),
    ("historic_nuclear", "Historic Nuclear", "rgba(126,87,194,0.6)"),
    ("new_nuclear", "New Nuclear", "rgba(194,94,166,0.6)"),
]
components = {metric: selected(metric) for metric, _, _ in component_specs}
components["other_market_steps"] = other_market_steps
active_components = [
    (metric, label, color)
    for metric, label, color in [
        *component_specs,
        ("other_market_steps", "Other Market Steps", "rgba(255,128,0,0.6)"),
    ]
    if has_energy(pd.DataFrame(components[metric]))
]

load_data = select_timeseries(load, weather_year)
dcline_flows = connected_dcline_flows(path=path, area=area, weather_year=weather_year)
net_ie = net_import_export(path, area, weather_year, load.index)

fig_gen = go.Figure()
for metric, label, color in active_components:
    fig_gen.add_trace(
        go.Scatter(
            x=load.index,
            y=components[metric],
            name=label,
            stackgroup="generation",
            fillcolor=color,
            line=dict(width=0),
        )
    )

if show_individual_lines:
    import_colors = [
        "rgba(128,0,128,0.6)",
        "rgba(255,0,255,0.6)",
        "rgba(180,0,180,0.6)",
        "rgba(200,50,200,0.6)",
    ]
    for idx, (line_name, flow_data) in enumerate(sorted(dcline_flows.items())):
        fig_gen.add_trace(
            go.Scatter(
                x=load.index,
                y=flow_data,
                name=f"Flow: {line_name}",
                stackgroup="generation",
                fillcolor=import_colors[idx % len(import_colors)],
                line=dict(width=0),
            )
        )
else:
    fig_gen.add_trace(
        go.Scatter(
            x=load.index,
            y=net_ie,
            name="Net Import (+) / Export (-)",
            stackgroup="generation",
            fillcolor="rgba(128,0,128,0.6)",
            line=dict(width=0),
        )
    )

fig_gen.add_trace(go.Scatter(x=load.index, y=load_data, name="Load", line=dict(color="red", width=3), mode="lines"))
fig_gen.add_hline(y=0, line=dict(color="black", width=1))
fig_gen.update_layout(
    title=f"Generation Mix vs Load - {area} - {weather_year}",
    xaxis_title="Time",
    yaxis_title="MW",
    hovermode="x unified",
    height=500,
)
st.plotly_chart(fig_gen, width="stretch")

rows = [{"Metric": "Load", "Value": float(load_data.mean())}]
for metric, label, _ in active_components:
    rows.append({"Metric": label, "Value": float(components[metric].mean())})
if show_individual_lines:
    for line_name, flow_data in sorted(dcline_flows.items()):
        rows.append({"Metric": f"Flow: {line_name}", "Value": float(flow_data.mean())})
else:
    rows.append({"Metric": "Net Import/Export", "Value": float(net_ie.mean())})
rows.append(
    {
        "Metric": "Total Generation + Import",
        "Value": float(sum(components[metric].mean() for metric, _, _ in active_components) + net_ie.mean()),
    }
)
st.dataframe(pd.DataFrame(rows).style.format({"Value": "{:.1f}"}), hide_index=True)


st.markdown("## DC Line Results")
if dc_records:
    dcline_options = [record["Line"] for record in dc_records]
    line_name = st.selectbox("Choose line:", dcline_options)
    line_flow = dcline_flow(path, line_name)
    fig_line = px.line(line_flow, title=f"DC Line {line_name}", labels={"index": "Time", "value": "Flow"})
    st.plotly_chart(fig_line, width="stretch")
else:
    st.info("No processed DC line records are available for this scenario.")


st.markdown("## Busbar Results")
if st.checkbox("Visualize busbar results", value=False):
    for metric, label, yaxis_title in [
        ("price", "Market Result Price", "EUR/MWh"),
        ("hydro", "Sum Hydro Production", "MW"),
        ("load", "Sum Load", "MW"),
        ("market_steps", "Sum Production from Market Steps", "MW"),
        ("historic_nuclear", "Historic Nuclear", "MW"),
        ("new_nuclear", "New Nuclear", "MW"),
        ("reservoir", "Reservoir", "Reservoir level"),
    ]:
        try:
            df_metric = busbar_metric(path, area, metric)
        except KeyError:
            continue
        df_plot = df_metric.copy()
        df_plot["MEAN"] = df_plot.mean(axis=1)
        fig = px.line(
            df_plot,
            title=f"{label} {area}",
            labels={"index": "Time", "value": yaxis_title},
        )
        st.plotly_chart(fig, width="stretch")


st.markdown("## Processed Reservoir Records")
scenario = load_processed_scenario(path)
reservoir_entities = scenario.get_reservoir_entities()
if st.checkbox("Visualize processed reservoir record", value=False):
    if not reservoir_entities:
        st.info("No processed individual reservoir records are available. Reprocess with --reservoir-mode all if needed.")
    else:
        reservoir_name = st.selectbox("Select reservoir:", reservoir_entities)
        metric = st.selectbox(
            "Metric:",
            ["reservoir_level", "reservoir_production", "reservoir_spill", "reservoir_discharge"],
        )
        try:
            df_reservoir = scenario.get_reservoir_metric(reservoir_name, metric)
        except KeyError:
            st.warning(f"No {metric} data found for {reservoir_name}.")
        else:
            df_plot = df_reservoir.copy()
            df_plot["MEAN"] = df_plot.mean(axis=1)
            fig = px.line(
                df_plot,
                title=f"{metric} - {reservoir_name}",
                labels={"index": "Time", "value": metric},
            )
            st.plotly_chart(fig, width="stretch")
