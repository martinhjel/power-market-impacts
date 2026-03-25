from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st
from lpr_sintef_bifrost.ltm import LTM
from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from app.utils.reservoirs import plot_reservoir, plot_reservoir_together

st.title("Input Data")

st.markdown("## input.h5 data")

# -- Sidebar: Result path selection --

paths = [p for p in (Path.cwd() / "ltm_output").glob("*/*") if p.is_dir()]
paths_dict = {str(i).split("/ltm_output/")[-1]: i  for i in paths}

selected_case = st.sidebar.selectbox("Select result:", paths_dict.keys(), index=0, key="sim1")
selected_path = paths_dict[selected_case]

st.sidebar.markdown(f"Using path: {selected_path}")


def get_elspot_area(df, elspot_area):
    ind = (df["omrType"] == "EL") & (df["omrnr"] == elspot_area)
    dff = df.loc[ind]
    dff = dff.set_index("dato_Id").sort_index()
    return dff

def get_nve_max_volume(df):
    ind = (df["omrType"] == "EL")
    dff = df.loc[ind, ["omrnr", "kapasitet_TWh"]]
    dff = dff.groupby("omrnr").max()
    dff["Elspot"] = [f"NO{i}" for i in dff.index]
    dff = dff.set_index("Elspot")
    return dff.to_dict()["kapasitet_TWh"]

def get_norway(df):
    dff = df.loc[(df["omrType"] == "NO")]
    return dff.set_index("dato_Id").sort_index()


def read_pyltmapi_model(path):
    return LTM.session_from_folder(path / "run_folder/emps")


df = pd.read_parquet("app/data/historic_reservoir_nve.parquet")
max_volume = get_nve_max_volume(df) # TWh

pyltm_session = read_pyltmapi_model(selected_path)
pyltm_model = pyltm_session.model

busbars = pyltm_model.busbars()
busbars_dict = {b.name: b for b in busbars}

data = []
for area in [f"NO{i}" for i in range(1, 6)]:
    max_volume_gwh = 0.0
    # for r in busbars_dict[area].reservoirs():
    #     max_volume_gwh += r.metadata["reservoir_capacity_mm3"] * r.metadata["global_energy_equivalent"]

    df_t = df_from_pyltm_result(busbars_dict[area].sum_reservoir()) / max_volume[area]/1e6
    df_t["omrType"] = "EL"
    df_t["omrnr"] = int(area[-1])
    data.append(df_t)

df_sim = pd.concat(data)
df_sim.columns = [1991 + i for i in range(30)] + ["omrType", "omrnr"]
df_sim["iso_uke"] = df_sim.index.isocalendar().week

df_sim = pd.melt(
    df_sim.reset_index(),
    id_vars=["index", "omrType", "omrnr", "iso_uke"],  # keep these columns
    var_name="iso_aar",
    value_name="value",
).rename(columns={"index": "dato_Id", "value": "fyllingsgrad"})

for area in range(1, 6):
    fig1 = plot_reservoir(get_elspot_area(df_sim, area))
    fig2 = plot_reservoir(get_elspot_area(df, area))

    combined_fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Simulated", "Actual"], shared_xaxes=True, shared_yaxes=False
    )

    # Add traces from fig1
    for trace in fig1.data:
        combined_fig.add_trace(deepcopy(trace), row=1, col=1)

    # Add traces from fig2
    for trace in fig2.data:
        combined_fig.add_trace(deepcopy(trace), row=1, col=2)

    combined_fig.update_layout(
        yaxis=dict(range=[0, 100]),
        yaxis2=dict(range=[0, 100]),
        height=500,
        width=1000,
        title_text=f"Reservoir Comparison NO{area}",
    )
    st.plotly_chart(combined_fig)

for area in range(1, 6):
    fig1 = plot_reservoir_together(dff_sim1=get_elspot_area(df_sim, area), dff_sim2=get_elspot_area(df, area), sim1_name="EMPS", sim2_name="Historical")
    
    fig1.update_layout(
        yaxis=dict(range=[0, 100]),
        height=500,
        width=1000,
        title_text=f"Reservoir Comparison NO {area}",
    )
    st.plotly_chart(fig1)
