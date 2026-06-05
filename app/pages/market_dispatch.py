from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.processed import (
    busbar_metric,
    busbar_names,
    connected_dcline_flows,
    default_index,
    list_processed_scenarios,
    load_processed_scenario,
    net_import_export,
    result_label,
    safe_busbar_metric,
    select_timeseries,
)

st.set_page_config(layout="wide")
st.title("Market Dispatch From Processed Results")

paths = list_processed_scenarios()
if not paths:
    st.error("No processed results found under ltm_processed/*/*/processed_data.parquet.")
    st.stop()

selected_path = st.sidebar.selectbox(
    "Select simulation:",
    paths,
    format_func=result_label,
    index=0,
)
path = str(selected_path)
load_processed_scenario(path)

st.sidebar.markdown(f"**Path:** `{selected_path}`")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

areas = busbar_names(path)
if not areas:
    st.error("No busbar records found in the processed data.")
    st.stop()

area = st.selectbox("Area", areas, index=default_index(areas, "NO2"))
load = busbar_metric(path, area, "load")
weather_years = ["Mean", *list(load.columns)]
weather_year = st.selectbox("Weather year", weather_years, index=0)


def metric_or_zero(metric: str) -> pd.DataFrame:
    return safe_busbar_metric(path, area, metric, like=load)


def series(metric: str) -> pd.Series:
    return select_timeseries(metric_or_zero(metric), weather_year)


def plot_timeseries(df: pd.DataFrame, title: str, columns: list[str], yaxis_title: str = "MW") -> go.Figure:
    fig = go.Figure()
    for column in columns:
        if column not in df.columns:
            continue
        fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column, mode="lines"))
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title=yaxis_title,
        legend_title="Series",
        height=500,
        hovermode="x unified",
    )
    return fig


fixed_nuclear = series("fixed_nuclear")
total_nuclear = series("total_nuclear")
flexible_nuclear_in_market_steps = total_nuclear.sub(fixed_nuclear, fill_value=0.0).clip(lower=0.0)
other_market_steps = series("market_steps").sub(flexible_nuclear_in_market_steps, fill_value=0.0)

export_df = pd.DataFrame(
    {
        "load": select_timeseries(load, weather_year),
        "hydro": series("hydro"),
        "onshore_wind": series("onshore_wind"),
        "offshore_wind": series("offshore_wind"),
        "solar": series("solar"),
        "historic_nuclear": series("historic_nuclear"),
        "new_nuclear": series("new_nuclear"),
        "other_market_steps": other_market_steps,
        "net_import_export": net_import_export(path, area, weather_year, load.index),
    }
)

process_df = pd.DataFrame(
    {
        "market_price": series("price"),
        "total_nuclear": series("total_nuclear"),
        "total_nuclear_available": series("total_nuclear_available"),
        "historic_nuclear": series("historic_nuclear"),
        "historic_nuclear_available": series("historic_nuclear_available"),
        "new_nuclear": series("new_nuclear"),
        "new_nuclear_available": series("new_nuclear_available"),
        "flexible_nuclear_in_market_steps": flexible_nuclear_in_market_steps,
        "other_market_steps": other_market_steps,
        "biomass": series("biomass"),
        "fossil_gas": series("fossil_gas"),
        "fossil_other": series("fossil_other"),
        "rationing": series("rationing"),
        "market_spillage": series("market_spillage"),
        "sum_market_steps": series("market_steps"),
    }
)
process_df["total_dispatch_plus_import"] = (
    export_df["hydro"]
    + export_df["onshore_wind"]
    + export_df["offshore_wind"]
    + export_df["solar"]
    + export_df["historic_nuclear"]
    + export_df["new_nuclear"]
    + export_df["other_market_steps"]
    + export_df["net_import_export"]
)
process_df["diff_to_load"] = process_df["total_dispatch_plus_import"] - export_df["load"]

left, right = st.columns(2)
with left:
    st.subheader("Dispatch Balance")
    st.plotly_chart(
        plot_timeseries(
            export_df,
            f"Dispatch: {area} | {weather_year}",
            [
                "load",
                "hydro",
                "onshore_wind",
                "offshore_wind",
                "solar",
                "historic_nuclear",
                "new_nuclear",
                "other_market_steps",
                "net_import_export",
            ],
        ),
        width="stretch",
    )

with right:
    st.subheader("Merit Order Reconstruction")
    st.plotly_chart(
        plot_timeseries(
            process_df,
            f"Technology Dispatch: {area} | {weather_year}",
            [
                "total_nuclear",
                "total_nuclear_available",
                "historic_nuclear",
                "historic_nuclear_available",
                "new_nuclear",
                "new_nuclear_available",
                "flexible_nuclear_in_market_steps",
                "other_market_steps",
                "biomass",
                "fossil_gas",
                "fossil_other",
                "rationing",
                "market_spillage",
                "sum_market_steps",
            ],
        ),
        width="stretch",
    )

st.subheader("Market Price")
st.plotly_chart(
    plot_timeseries(process_df, f"Market Price: {area} | {weather_year}", ["market_price"], yaxis_title="EUR/MWh"),
    width="stretch",
)

if st.checkbox("Show individual DC line flows", value=False):
    flow_df = pd.DataFrame(connected_dcline_flows(path=path, area=area, weather_year=weather_year))
    if flow_df.empty:
        st.info("No connected DC line flow records found for this area.")
    else:
        st.plotly_chart(
            plot_timeseries(flow_df, f"Connected DC Lines: {area} | {weather_year}", list(flow_df.columns)),
            width="stretch",
        )

st.subheader("Averages")
avg_df = pd.concat(
    [
        export_df.mean().rename("dispatch_avg_mw"),
        process_df.mean().rename("process_avg"),
    ],
    axis=1,
)
st.dataframe(avg_df.style.format("{:.2f}"))
