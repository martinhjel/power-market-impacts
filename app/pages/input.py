from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import h5py
from app.utils.maps import plot_grid_on_map
from data import PowerGamaDataLoader
from lpr_sintef_bifrost.utils.time import CET_winter
from lpr_sintef_bifrost.ltm import LTM
from app.utils.market_steps import get_market_steps
st.title("Input Data")

dataset_year = 2025
dataset_scenario = "BM"
dataset_version = "100"
base_path = Path("/mnt/NordicNuclearAnalysis")
combined = True
dataset_path = base_path / f"CASE_{dataset_year}/scenario_{dataset_scenario}/data/system"


#%%

st.markdown("## input.h5 data")

# -- Sidebar: Result path selection --
result_paths = [i for i in (Path.cwd() / "ltm_output").glob("*/*") if i.is_dir()]
selected_path = st.sidebar.selectbox("Select results:", result_paths)
st.sidebar.markdown(f"Using path: {selected_path}")

# Store path in session state to share across pages or re-renders
st.session_state["path"] = selected_path
st.write(f"Selected path: {selected_path}")

@st.cache_resource
def load_price_series_data(path: Path, area: str) -> pd.DataFrame:
    """Load price series data from input.h5 for given area."""
    with h5py.File(path / "results/input.h5", "r") as h5file:
        time = h5file[f"/price_series_secondary/{area}_PRICE_SERIES/series/times"][:]
        value = h5file[f"/price_series_secondary/{area}_PRICE_SERIES/series/vals"][:]
        df = pd.DataFrame(value, index=time)
        df.index = pd.to_datetime(df.index, unit="ms").tz_localize(CET_winter)
    return df

# -- Area selection --
price_areas = ["DE", "GB", "PL"]
price_area = st.selectbox("Select price area:", price_areas, index=0)

# -- Plot price series data --
try:
    df = load_price_series_data(selected_path, price_area)
    fig = px.line(df.resample("1D").mean(), title=f"{price_area} Price Series - 1D resampled")
    st.plotly_chart(fig, use_container_width=True)
except KeyError:
    st.error(f"Price series data for area '{price_area}' not found in {selected_path}/results/input.h5")

@st.cache_resource
def load_wind_groups(path: Path) -> dict:
    """Scan and find all wind_on capacity groups in input.h5."""
    with h5py.File(path / "results/input.h5", "r") as h5file:
        wind_groups = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and "wind_on/capacity/times" in name:
                group_name = name.split("/wind_on")[0]
                wind_groups.append(group_name)
        h5file.visititems(visitor)
        wind_groups = sorted(set(wind_groups))
        if not wind_groups:
            st.error("No wind_on capacity data found in the file.")
            st.stop()
        # Map area (e.g., 'SE4') -> full group path
        return {group.split("/")[1].split(" ")[0]: group for group in wind_groups}

wind_groups_dict = load_wind_groups(selected_path)

# --- Let user choose areas to plot ---
selected_area = st.selectbox("Choose wind areas:", wind_groups_dict.keys(), index=0)
if not selected_area:
    st.warning("Please select one area to plot.")
    st.stop()

@st.cache_resource
def load_wind_data(path, area):
    with h5py.File(path /"results/input.h5", "r") as h5file:
        # Load times & values for each selected area
        times_ds = h5file[wind_groups_dict[area]]
        vals_ds = h5file[wind_groups_dict[area].replace("times","vals")]

        times = pd.to_datetime(times_ds[:], unit="ms").tz_localize("UTC")
        vals = vals_ds[:]

        df = pd.DataFrame(vals, index=times)
    return df

df = load_wind_data(selected_path, selected_area)

fig = px.line(
    df.resample("1D").mean(),
    labels={"value": "Capacity (MW)", "index": "Time"},
    title="Wind Onshore Capacity - Resampled 1D",
)
st.plotly_chart(fig, use_container_width=True)

#%%

data_loader = PowerGamaDataLoader(
    year=dataset_year, scenario=dataset_scenario, version=dataset_version, base_path=base_path, combined=combined
)
# NB: Temporary overwriting of generator data
generator = pd.read_csv("data/generator.csv", index_col=0)
data_loader.generator = generator

profiles_file = Path("/mnt/Profiler/timeseries_profiles_v3.csv")

@st.cache_data
def get_profiles(profiles_file):
    return pd.read_csv(profiles_file, index_col=0, parse_dates=True)

# df_profiles = get_profiles(profiles_file)

df_renewables_profiles = pd.read_parquet("data/renewables_profiles.parquet")
df_load_profiles = pd.read_parquet("data/load_profiles.parquet")

db_profiles = set(list(df_renewables_profiles.columns) + list(df_load_profiles.columns))
gen_profiles = set(data_loader.generator["inflow_ref"].unique())

st.text("Available profiles from csv db:")
st.text(db_profiles)

st.text("Generator profiles:")
st.text(gen_profiles)

st.markdown("## Generators")
st.markdown("The generator file from PowerGama, which may not be equal to the market steps that are added to the EMPS model.")
nodes = data_loader.generator["node"].unique()
selected_nodes = st.multiselect("Select nodes: ", nodes, default=nodes[0])
st.dataframe(data_loader.generator.loc[data_loader.generator["node"].isin(selected_nodes)])

st.markdown("### Market Steps")
st.markdown("The actual market steps added to the EMPS model.")
df_market_steps = get_market_steps(selected_path=selected_path, selected_nodes=selected_nodes)
st.dataframe(df_market_steps)

ren_list = list(df_renewables_profiles)
ren_list.sort()
ren_select = st.multiselect("Select renewable profiles:", ren_list, default=ren_list[:1])

fig = go.Figure()
years = st.multiselect("Select years: ", df_renewables_profiles.index.year.unique(), default=2020)
fig = px.line(df_renewables_profiles.loc[df_renewables_profiles.index.year.isin(years), ren_select])
st.plotly_chart(fig)

load_list = data_loader.consumer["demand_ref"].unique()
load_select = st.multiselect("Select load profiles: ", load_list, default=load_list[:1])
fig = px.line(df_load_profiles.loc[df_load_profiles.index.year.isin(years), load_select])
st.plotly_chart(fig)

st.markdown("## Load data")
st.dataframe(data_loader.consumer)

st.markdown("## Lines")
df_links = data_loader.branch
st.dataframe(df_links)

nodes_csv = Path.cwd() / "app/data/nodes_location.csv"
df_nodes = df_nodes = pd.read_csv(nodes_csv, index_col="id")

fig = plot_grid_on_map(df_links, df_nodes)
st.plotly_chart(fig)


# try:
#     # results = load_results(path)

#     with h5py.File(path / "results/results.h5", "r") as h5file:
#         dataset_paths = list_hdf5_datasets(h5file)
#         selected_path = st.selectbox("Select dataset to view", dataset_paths)

#         data = read_hdf5_dataset(h5file, selected_path)
#         st.write(f"Shape: {data.shape}, Dtype: {data.dtype}")

# except Exception as e:
#     st.error(f"Failed to load results: {e}")
#     st.stop()
