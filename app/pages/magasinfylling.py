from __future__ import annotations

from copy import deepcopy

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

from app.utils.processed import (
    busbar_metric,
    busbar_names,
    list_processed_scenarios,
    load_processed_scenario,
    result_label,
)
from app.utils.reservoirs import plot_reservoir, plot_reservoir_together

st.set_page_config(layout="wide")
st.title("Magasinfylling")


@st.cache_data(show_spinner=False)
def read_historic_reservoir() -> pd.DataFrame:
    return pd.read_parquet("app/data/historic_reservoir_nve.parquet")


def get_elspot_area(df: pd.DataFrame, elspot_area: int) -> pd.DataFrame:
    ind = (df["omrType"] == "EL") & (df["omrnr"] == elspot_area)
    dff = df.loc[ind]
    return dff.set_index("dato_Id").sort_index()


def get_nve_max_volume(df: pd.DataFrame) -> dict[str, float]:
    ind = df["omrType"] == "EL"
    dff = df.loc[ind, ["omrnr", "kapasitet_TWh"]]
    dff = dff.groupby("omrnr").max()
    dff["Elspot"] = [f"NO{i}" for i in dff.index]
    dff = dff.set_index("Elspot")
    return dff.to_dict()["kapasitet_TWh"]


def reservoir_matrix_to_nve_frame(
    reservoir: pd.DataFrame,
    *,
    area: str,
    max_volume_twh: float,
    first_weather_year: int = 1991,
) -> pd.DataFrame:
    dff = reservoir / max_volume_twh / 1e6
    dff = dff.copy()
    dff.columns = [first_weather_year + idx for idx, _ in enumerate(dff.columns)]
    dff["omrType"] = "EL"
    dff["omrnr"] = int(area[-1])
    dff["iso_uke"] = dff.index.isocalendar().week.astype(int)
    dff.index.name = "dato_Id"

    return pd.melt(
        dff.reset_index(),
        id_vars=["dato_Id", "omrType", "omrnr", "iso_uke"],
        var_name="iso_aar",
        value_name="fyllingsgrad",
    )


paths = list_processed_scenarios()
if not paths:
    st.error("No processed results found under ltm_processed/*/*/processed_data.parquet.")
    st.stop()

selected_path = st.sidebar.selectbox(
    "Select result:",
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

historic = read_historic_reservoir()
max_volume = get_nve_max_volume(historic)
available_areas = [area for area in [f"NO{i}" for i in range(1, 6)] if area in busbar_names(path)]

if not available_areas:
    st.error("No Norwegian reservoir busbar records found in the processed data.")
    st.stop()

selected_areas = st.multiselect(
    "Areas",
    available_areas,
    default=available_areas,
)
if not selected_areas:
    st.info("Select at least one area.")
    st.stop()

data = []
missing = []
for area in selected_areas:
    try:
        reservoir = busbar_metric(path, area, "reservoir")
    except KeyError:
        missing.append(area)
        continue
    data.append(
        reservoir_matrix_to_nve_frame(
            reservoir,
            area=area,
            max_volume_twh=max_volume[area],
        )
    )

if missing:
    st.warning(f"Missing processed reservoir data for: {', '.join(missing)}")
if not data:
    st.stop()

df_sim = pd.concat(data, ignore_index=True)

area_tabs = st.tabs(selected_areas)
for tab, area in zip(area_tabs, selected_areas):
    area_number = int(area[-1])
    with tab:
        fig_sim = plot_reservoir(get_elspot_area(df_sim, area_number))
        fig_actual = plot_reservoir(get_elspot_area(historic, area_number))

        combined_fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Simulated", "Actual"],
            shared_xaxes=True,
            shared_yaxes=False,
        )

        for trace in fig_sim.data:
            combined_fig.add_trace(deepcopy(trace), row=1, col=1)
        for trace in fig_actual.data:
            combined_fig.add_trace(deepcopy(trace), row=1, col=2)

        combined_fig.update_layout(
            yaxis=dict(range=[0, 100]),
            yaxis2=dict(range=[0, 100]),
            height=500,
            title_text=f"Reservoir Comparison {area}",
        )
        st.plotly_chart(combined_fig, width="stretch")

        together = plot_reservoir_together(
            dff_sim1=get_elspot_area(df_sim, area_number),
            dff_sim2=get_elspot_area(historic, area_number),
            sim1_name="EMPS",
            sim2_name="Historical",
        )
        together.update_layout(
            yaxis=dict(range=[0, 100]),
            height=500,
            title_text=f"Reservoir Percentiles {area}",
        )
        st.plotly_chart(together, width="stretch")
