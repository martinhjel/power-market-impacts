"""
Get renewables capacity time series from LTM model.
"""

from pathlib import Path

import pandas as pd
from lpr_sintef_bifrost.ltm import LTM
from pyltm.pyltm import LtmApiModel

scenario_name = "test_FalseHYD_FalseFF_BALOAD_20.00TWH_NoneNUKE_NoneOFF"
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
result_path = Path.cwd() / "ltm_output" / MODEL_FOLDER / scenario_name / "run_folder/emps/"

# Load from results directory to get actual simulation output (hourly resolution)
pyltm_session = LTM.session_from_folder(result_path)
busbars = {b.name: b for b in pyltm_session.model.busbars()}

model = pyltm_session.model


class Renewables:
    def __init__(self, model: LtmApiModel) -> None:
        self.model = model

        solar_dict = {str(i).split("_")[1]: i for i in model.solar()}

        off_wind_dict = {}
        on_wind_dict = {}
        for w in model.wind():
            if "_off" in str(w):
                off_wind_dict[str(w).split("_")[1]] = w
            elif "_on" in str(w):
                on_wind_dict[str(w).split("_")[1]] = w

        self.solar = solar_dict
        self.off_wind = off_wind_dict
        self.on_wind = on_wind_dict

    def get_onshore_wind_capacity(self, area: str):
        data = self.on_wind[area].capacity.scenarios
        ind = self.on_wind[area].capacity.timestamps

        return pd.DataFrame(data, columns=ind).T

    def get_offshore_wind_capacity(self, area: str):
        data = self.off_wind[area].capacity.scenarios
        ind = self.off_wind[area].capacity.timestamps

        return pd.DataFrame(data, columns=ind).T

    def get_solar_capacity(self, area: str):
        data = self.solar[area].capacity.scenarios
        ind = self.solar[area].capacity.timestamps

        return pd.DataFrame(data, columns=ind).T


from matplotlib import pyplot as plt

renewables = Renewables(model)
renewables.solar.keys()
for area in renewables.solar.keys():
    try:
        df_off_wind = renewables.get_offshore_wind_capacity(area)
        df_off_wind.mean(axis=1).plot(title=area)
        df_solar = renewables.get_solar_capacity(area)
        df_solar.mean(axis=1).plot()
        plt.show()
    except Exception as e:
        print(f"Failed for {area}: {e}")

df_renewables_profiles = pd.read_parquet("data/renewables_profiles.parquet")
df_renewables_profiles["SE1_wind_onshore_new"].mean() * 3000
df_renewables_profiles.columns


df_solar = renewables.get_solar_capacity("DK2")
df_solar.mean(axis=1).plot()
df_solar.plot()

df_on_wind = renewables.get_onshore_wind_capacity("DK2")
df_on_wind.mean(axis=1).plot()

[i for i in dir(busbars["NO2"]) if "__" not in i]
