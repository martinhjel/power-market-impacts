from __future__ import annotations

from pathlib import Path

import pandas as pd
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
    result_label,
    safe_busbar_metric,
    select_timeseries,
)

st.set_page_config(layout="wide")
st.title("EMPS Processed Result Comparison")

paths = list_processed_scenarios()
if not paths:
    st.error("No processed results found under ltm_processed/*/*/processed_data.parquet.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    sim1_path = st.selectbox("Select first simulation:", paths, format_func=result_label, index=0, key="sim1")
with col2:
    sim2_index = 1 if len(paths) > 1 else 0
    sim2_path = st.selectbox("Select second simulation:", paths, format_func=result_label, index=sim2_index, key="sim2")

sim1 = str(sim1_path)
sim2 = str(sim2_path)
load_processed_scenario(sim1)
load_processed_scenario(sim2)

st.sidebar.write("**Sim1:**")
st.sidebar.write(result_label(sim1_path))
st.sidebar.write("**Sim2:**")
st.sidebar.write(result_label(sim2_path))
if st.sidebar.button("Reload processed data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

nodes_csv = Path.cwd() / "app/data/nodes_location.csv"
df_nodes = pd.read_csv(nodes_csv, index_col="id")


@st.cache_data(show_spinner="Computing weighted prices...")
def compute_avg_weighted_prices(path: str) -> dict[str, float]:
    prices: dict[str, float] = {}
    for area in busbar_names(path):
        try:
            price = busbar_metric(path, area, "price")
            load = busbar_metric(path, area, "load")
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


def has_energy(df: pd.DataFrame) -> bool:
    return bool(df.abs().sum().sum() > 1e-6)


def metric_or_zero(path: str, area: str, metric: str, like: pd.DataFrame) -> pd.DataFrame:
    return safe_busbar_metric(path, area, metric, like=like)


def add_matrix_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    name: str,
    color: str,
    show_distribution: bool,
) -> None:
    if show_distribution:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.max(axis=1),
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.min(axis=1),
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                fillcolor=color.replace("1)", "0.18)"),
                name=f"{name} min-max",
            )
        )
        fig.add_trace(go.Scatter(x=df.index, y=df.mean(axis=1), name=f"{name} mean", line=dict(color=color)))
        fig.add_trace(
            go.Scatter(x=df.index, y=df.median(axis=1), name=f"{name} median", line=dict(color=color, dash="dot"))
        )
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df.mean(axis=1), name=name, line=dict(color=color)))


def plot_matrix_comparison(
    *,
    title: str,
    yaxis_title: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    show_distribution: bool,
) -> None:
    fig = go.Figure()
    add_matrix_trace(fig, df1, name="Sim 1", color="rgba(30, 92, 210, 1)", show_distribution=show_distribution)
    add_matrix_trace(fig, df2, name="Sim 2", color="rgba(210, 73, 45, 1)", show_distribution=show_distribution)
    suffix = " min/max/mean/median" if show_distribution else " mean"
    fig.update_layout(title=f"{title} ({suffix})", xaxis_title="Time", yaxis_title=yaxis_title, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


price_map_1 = compute_avg_weighted_prices(sim1)
price_map_2 = compute_avg_weighted_prices(sim2)
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

st.markdown("## Price Difference")
if df_price_nodes.empty:
    st.info("No overlapping mapped nodes found between the selected simulations.")
else:
    diff_table = df_price_compare.reset_index().rename(
        columns={
            "index": "Node",
            "avg_weighted_price_sim1": "Sim1 Avg Weighted Price",
            "avg_weighted_price_sim2": "Sim2 Avg Weighted Price",
            "delta_price": "Delta Price (Sim2 - Sim1)",
        }
    )
    delta_max = diff_table["Delta Price (Sim2 - Sim1)"].abs().max()
    style = diff_table.style.format(
        {
            "Sim1 Avg Weighted Price": "{:.2f}",
            "Sim2 Avg Weighted Price": "{:.2f}",
            "Delta Price (Sim2 - Sim1)": "{:.2f}",
        }
    )
    if pd.notna(delta_max) and delta_max > 0:
        style = style.background_gradient(
            cmap="RdBu_r",
            subset=["Delta Price (Sim2 - Sim1)"],
            vmin=-delta_max,
            vmax=delta_max,
        )
    st.dataframe(style, hide_index=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=df_price_nodes["lon"],
            lat=df_price_nodes["lat"],
            mode="markers+text",
            text=[
                f"<b>{node}</b><br>Sim1: {row['avg_weighted_price_sim1']:.1f} EUR/MWh"
                f"<br>Sim2: {row['avg_weighted_price_sim2']:.1f} EUR/MWh"
                f"<br>Delta: {row['delta_price']:.1f} EUR/MWh"
                for node, row in df_price_nodes.iterrows()
            ],
            textposition="top center",
            marker=dict(
                size=12,
                color=df_price_nodes["delta_price"],
                colorscale="RdBu_r",
                cmid=0,
                colorbar=dict(title="Delta price"),
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
        title_text="Average Weighted Price Difference (Sim2 - Sim1)",
        height=600,
    )
    st.plotly_chart(fig, width="stretch")

common_busbars = sorted(set(busbar_names(sim1)).intersection(busbar_names(sim2)))
if not common_busbars:
    st.error("The selected processed scenarios have no common busbars.")
    st.stop()

busbar_name = st.selectbox(
    "Select busbar to compare:",
    common_busbars,
    index=default_index(common_busbars, "NO2"),
)

df_load_1 = busbar_metric(sim1, busbar_name, "load")
df_load_2 = busbar_metric(sim2, busbar_name, "load")
weather_years = sorted(set(df_load_1.columns).intersection(df_load_2.columns))

use_distribution = st.checkbox("Show min/max/mean/median across weather years", value=False)
col_s1, col_s2, col_display = st.columns(3)
with col_s1:
    scenario_1 = st.selectbox("Weather year for Sim 1:", ["Mean", *weather_years], key="scenario_1")
with col_s2:
    scenario_2 = st.selectbox("Weather year for Sim 2:", ["Mean", *weather_years], key="scenario_2")
with col_display:
    show_individual_lines = st.checkbox("Show individual DC lines", value=False)

st.header(f"Generation Mix and Net Import/Export for {busbar_name}")

component_specs = [
    ("hydro", "Hydro", "rgba(0,100,255,0.6)"),
    ("onshore_wind", "Onshore Wind", "rgba(100,190,100,0.6)"),
    ("offshore_wind", "Offshore Wind", "rgba(50,145,50,0.6)"),
    ("solar", "Solar", "rgba(245,190,0,0.6)"),
    ("historic_nuclear", "Historic Nuclear", "rgba(126,87,194,0.6)"),
    ("new_nuclear", "New Nuclear", "rgba(194,94,166,0.6)"),
    ("biomass", "Biomass", "rgba(95,120,50,0.6)"),
    ("fossil_gas", "Fossil Gas", "rgba(180,120,50,0.6)"),
    ("fossil_other", "Fossil Other", "rgba(120,120,120,0.6)"),
    ("rationing", "Rationing", "rgba(210,40,40,0.6)"),
]

component_matrices_1 = {
    metric: metric_or_zero(sim1, busbar_name, metric, df_load_1) for metric, _, _ in component_specs
}
component_matrices_2 = {
    metric: metric_or_zero(sim2, busbar_name, metric, df_load_2) for metric, _, _ in component_specs
}

active_components = [
    (metric, label, color)
    for metric, label, color in component_specs
    if has_energy(component_matrices_1[metric]) or has_energy(component_matrices_2[metric])
]

series_1 = {
    metric: select_timeseries(component_matrices_1[metric], scenario_1) for metric, _, _ in active_components
}
series_2 = {
    metric: select_timeseries(component_matrices_2[metric], scenario_2) for metric, _, _ in active_components
}
load_1 = select_timeseries(df_load_1, scenario_1)
load_2 = select_timeseries(df_load_2, scenario_2)

dcline_flows_1 = connected_dcline_flows(path=sim1, area=busbar_name, weather_year=scenario_1)
dcline_flows_2 = connected_dcline_flows(path=sim2, area=busbar_name, weather_year=scenario_2)
net_ie_1 = net_import_export(sim1, busbar_name, scenario_1, df_load_1.index)
net_ie_2 = net_import_export(sim2, busbar_name, scenario_2, df_load_2.index)


def plot_generation_mix(
    *,
    title: str,
    index: pd.Index,
    load: pd.Series,
    components: dict[str, pd.Series],
    dcline_flows: dict[str, pd.Series],
    net_ie: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    for metric, label, color in active_components:
        fig.add_trace(
            go.Scatter(
                x=index,
                y=components[metric],
                name=label,
                stackgroup="generation",
                fillcolor=color,
                line=dict(width=0),
            )
        )
    if show_individual_lines:
        colors = [
            "rgba(128,0,128,0.6)",
            "rgba(255,0,255,0.6)",
            "rgba(180,0,180,0.6)",
            "rgba(200,50,200,0.6)",
        ]
        for idx, (line_name, flow_data) in enumerate(sorted(dcline_flows.items())):
            fig.add_trace(
                go.Scatter(
                    x=index,
                    y=flow_data,
                    name=f"Flow: {line_name}",
                    stackgroup="generation",
                    fillcolor=colors[idx % len(colors)],
                    line=dict(width=0),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=index,
                y=net_ie,
                name="Net Import (+) / Export (-)",
                stackgroup="generation",
                fillcolor="rgba(128,0,128,0.6)",
                line=dict(width=0),
            )
        )
    fig.add_trace(go.Scatter(x=index, y=load, name="Load", line=dict(color="red", width=3), mode="lines"))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="MW", hovermode="x unified", height=500)
    return fig


left, right = st.columns(2)
with left:
    st.plotly_chart(
        plot_generation_mix(
            title=f"Sim 1 - {scenario_1}",
            index=df_load_1.index,
            load=load_1,
            components=series_1,
            dcline_flows=dcline_flows_1,
            net_ie=net_ie_1,
        ),
        width="stretch",
    )
with right:
    st.plotly_chart(
        plot_generation_mix(
            title=f"Sim 2 - {scenario_2}",
            index=df_load_2.index,
            load=load_2,
            components=series_2,
            dcline_flows=dcline_flows_2,
            net_ie=net_ie_2,
        ),
        width="stretch",
    )

st.subheader("Average Values (MW)")
rows = []
for metric, label, _ in active_components:
    rows.append(
        {
            "Metric": label,
            "Sim 1": float(series_1[metric].mean()),
            "Sim 2": float(series_2[metric].mean()),
        }
    )
rows.append({"Metric": "Load", "Sim 1": float(load_1.mean()), "Sim 2": float(load_2.mean())})
if show_individual_lines:
    for line_name in sorted(set(dcline_flows_1) | set(dcline_flows_2)):
        rows.append(
            {
                "Metric": f"Flow: {line_name}",
                "Sim 1": float(dcline_flows_1.get(line_name, pd.Series(0.0, index=df_load_1.index)).mean()),
                "Sim 2": float(dcline_flows_2.get(line_name, pd.Series(0.0, index=df_load_2.index)).mean()),
            }
        )
else:
    rows.append({"Metric": "Net Import/Export", "Sim 1": float(net_ie_1.mean()), "Sim 2": float(net_ie_2.mean())})

summary = pd.DataFrame(rows)
summary["Difference (Sim2 - Sim1)"] = summary["Sim 2"] - summary["Sim 1"]
st.dataframe(
    summary.style.format({"Sim 1": "{:.1f}", "Sim 2": "{:.1f}", "Difference (Sim2 - Sim1)": "{:.1f}"}),
    hide_index=True,
)

st.markdown("## Market Step Reconstruction")
market_metrics = [
    "market_steps",
    "total_nuclear",
    "total_nuclear_available",
    "historic_nuclear",
    "historic_nuclear_available",
    "new_nuclear",
    "new_nuclear_available",
    "biomass",
    "fossil_gas",
    "fossil_other",
    "rationing",
    "market_spillage",
]
market_rows = []
for metric in market_metrics:
    m1 = metric_or_zero(sim1, busbar_name, metric, df_load_1)
    m2 = metric_or_zero(sim2, busbar_name, metric, df_load_2)
    market_rows.append(
        {
            "Metric": metric,
            "Sim 1 avg MW": float(m1.mean().mean()),
            "Sim 2 avg MW": float(m2.mean().mean()),
            "Difference": float(m2.mean().mean() - m1.mean().mean()),
        }
    )
st.dataframe(
    pd.DataFrame(market_rows).style.format(
        {"Sim 1 avg MW": "{:.1f}", "Sim 2 avg MW": "{:.1f}", "Difference": "{:.1f}"}
    ),
    hide_index=True,
)

st.header(f"Price Comparison for {busbar_name}")
df_price_1 = busbar_metric(sim1, busbar_name, "price")
df_price_2 = busbar_metric(sim2, busbar_name, "price")
plot_matrix_comparison(
    title="Market Price",
    yaxis_title="EUR/MWh",
    df1=df_price_1,
    df2=df_price_2,
    show_distribution=use_distribution,
)

col1, col2 = st.columns(2)
with col1:
    st.write("Sim 1")
    st.write(f"Average price: {df_price_1.mean().mean():.2f} EUR/MWh")
    st.write(f"Average std. dev.: {df_price_1.std().mean():.2f}")
with col2:
    st.write("Sim 2")
    st.write(f"Average price: {df_price_2.mean().mean():.2f} EUR/MWh")
    st.write(f"Average std. dev.: {df_price_2.std().mean():.2f}")

hist = go.Figure()
hist.add_trace(go.Histogram(x=df_price_1.to_numpy().ravel(), name="Sim 1", opacity=0.55))
hist.add_trace(go.Histogram(x=df_price_2.to_numpy().ravel(), name="Sim 2", opacity=0.55))
hist.update_layout(barmode="overlay", title="Price Distribution Histogram", xaxis_title="EUR/MWh")
st.plotly_chart(hist, width="stretch")

st.header("Transmission Comparison")


@st.cache_data(show_spinner="Computing mean flows...")
def compute_mean_flows(path: str) -> pd.DataFrame:
    rows = []
    for name in dcline_names(path):
        flow = dcline_flow(path, name)
        rows.append({"name": name, "flow": float(flow.mean().mean())})
    return pd.DataFrame(rows).set_index("name")


df_line1 = compute_mean_flows(sim1)
df_line2 = compute_mean_flows(sim2)
df_line = pd.concat([df_line1.rename(columns={"flow": "Sim1"}), df_line2.rename(columns={"flow": "Sim2"})], axis=1)
df_line["Sim1-Sim2"] = df_line["Sim1"] - df_line["Sim2"]
st.dataframe(df_line.style.format("{:.1f}").background_gradient(cmap="coolwarm"))

st.header(f"Hydro Production Comparison for {busbar_name}")
try:
    df_hydro_1 = busbar_metric(sim1, busbar_name, "hydro")
    df_hydro_2 = busbar_metric(sim2, busbar_name, "hydro")
    plot_matrix_comparison(
        title="Hydro Production",
        yaxis_title="MW",
        df1=df_hydro_1,
        df2=df_hydro_2,
        show_distribution=use_distribution,
    )
except KeyError:
    st.warning("Hydro production data missing for selected busbar.")

st.header(f"Load Comparison for {busbar_name}")
plot_matrix_comparison(
    title="Load",
    yaxis_title="MW",
    df1=df_load_1,
    df2=df_load_2,
    show_distribution=use_distribution,
)

st.header(f"Reservoir Comparison for {busbar_name}")
try:
    df_res_1 = busbar_metric(sim1, busbar_name, "reservoir")
    df_res_2 = busbar_metric(sim2, busbar_name, "reservoir")
    plot_matrix_comparison(
        title="Reservoir",
        yaxis_title="Reservoir level",
        df1=df_res_1,
        df2=df_res_2,
        show_distribution=use_distribution,
    )
except KeyError:
    st.warning("Reservoir data missing for selected busbar.")
