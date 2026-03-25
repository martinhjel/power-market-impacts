#%%
from pathlib import Path
import h5py
from lpr_sintef_bifrost.results import LTMSessionResults
from lpr_sintef_bifrost.models import EMPSConfigurationBuilder
from lpr_sintef_bifrost.ltm import LTM
from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result
from pyltmapi import LtmPlot
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline # For jupyter
#%%

from IPython.display import HTML, display
import pyltm
import pandas as pd
from importlib.metadata import version

def show_ltmapi_attributes():
    attributes = pyltm.LtmApiModule.attributes()
    frame = pd.DataFrame(attributes)
    frame.to_csv("pyltm_attributes.csv")
    display(HTML(frame.to_html()))

display(f"pyltm {version("pyltm")}")

show_ltmapi_attributes()

#%%

paths = [i for i in (Path.cwd()/"ltm_output").glob("*")]
path = paths[-1]

pyltm_session = LTM.session_from_folder(path / "run_folder/emps")
pyltm_model = pyltm_session.model

busbars = pyltm_model.busbars()
busbars_dict = {b.name: b for b in busbars}

agg_hydros = pyltm_model.aggregated_hydro_modules()

agg_hydro = agg_hydros[0]
agg_hydro.name

market_step = pyltm_model.market_steps()[0]

market_step_data = []
busbar_name = "NO2"
for market_step in pyltm_model.market_steps():
    if busbar_name == market_step.busbar_name:
        market_step_data.append({
            "capacity": market_step.capacity.scenarios,
            "name": market_step.name,
            "price": getattr(market_step.price, "scenarios", market_step.price),
        })


pd.DataFrame(market_step_data)
#%%

# %% Read results

resolution = "3H"
run_folder = Path().cwd() / f"ltm_output/test_{resolution}"

config = EMPSConfigurationBuilder.from_result_folder(run_folder)
config._result_session = LTM.session_from_folder(run_folder/"run_folder/emps")
pyltm_session = config._result_session
pyltm_model = pyltm_session.model
#%%

pyltm_model.dclines()
for dcline in pyltm_model.dclines():
    dc_line = df_from_pyltm_result(dcline.transmission_results())
    dc_line.plot()

#%%

def txy_to_frame(txy):
    return pd.DataFrame(txy.scenarios, columns=txy.timestamps).T

def agg_reservoir_to_frame(busbar):
    time, values = busbar.reservoir()
    time = np.array(time, dtype="datetime64[ms]")
    values = np.array(values, copy=False)
    return pd.DataFrame(values.T, index=time)

pyltm_model.aggregated_hydro_modules()[0].regulated_energy_inflow.timestamps

df = txy_to_frame(pyltm_model.aggregated_hydro_modules()[0].regulated_energy_inflow)
df.plot()

df = txy_to_frame(pyltm_model.aggregated_hydro_modules()[0].station_power)
df.plot()

busbar= busbars_dict["FI"]
df = agg_reservoir_to_frame(busbar)
df.plot()

#%% # Average hydro produciton in finland is 14 600 GWh
df_renewables_profiles = pd.read_parquet("data/renewables_profiles.parquet")
generator = pd.read_csv("data/generator.csv", index_col=0)
generator.loc[(generator["type"].isin(["hydro", "ror"]))].T
generator.loc[(generator["node"] == "FI") & (generator["type"].isin(["hydro", "ror"]))].T
inflow_ref = "inflow_NO1"
p_max = 344 #MW
inflow_fac = 0.46
storage_cap = 5.954053e5 # MWh
df_temp = (df_renewables_profiles[inflow_ref] * p_max * inflow_fac).to_frame()
df_temp.plot()

(df_temp.resample("1Y").mean()*8760).mean()/1e3

#%%

LtmPlot.make_generic_plot()

busbars = pyltm_model.busbars()

dcline = pyltm_model.dclines()[0]

#%%

config.visualize()


#%%

df = pd.read_parquet("volt_price_forecast_de.parquet")

df.plot()
import matplotlib.pyplot as plt
df[df.index.year == 2025].plot()
import plotly.express as px
px.line(df[df.index.year == 2025])


from entsoe import EntsoePandasClient
import pandas as pd

import configparser
config = configparser.ConfigParser()
config_file = Path("config.ini")
config_file.exists()
config.read(config_file)

api_key = config["entsoe"]["security_token"]
client = EntsoePandasClient(api_key=api_key)

from lpr_sintef_bifrost.utils.time import CET_winter

country_codes = ["DE_LU","PL"]
start = pd.Timestamp('20240101', tz=CET_winter)
end = pd.Timestamp('20250101', tz=CET_winter)

for country_code in country_codes:
    country_code = "GB_IFA2"
    df = client.query_day_ahead_prices(country_code, start=start, end=end)
    df.to_csv(Path.cwd()/ f"data/day_ahead_{country_code}_{start}_{end}.csv")
    
