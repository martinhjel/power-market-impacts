# %%


import configparser

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from entsoe import EntsoePandasClient
from lpr_sintef_bifrost.utils.time import CET_winter

from layout import my_layout

# %% Helpers

config = configparser.ConfigParser()
config.read("config.ini")

web_token = config["entsoe"]["security_token"]

# %%
client = EntsoePandasClient(api_key=web_token)


# %%

start = pd.Timestamp("2015-01-01T00:00Z")
end = pd.Timestamp("2025-01-01T00:00Z")
country_code = "SE"
psr_type = "B14"



df = client.query_generation(country_code=country_code, start=start, end=end, psr_type=psr_type)

df.to_parquet("data/swedish_nuclear.parquet")
# df = pd.read_parquet("data/swedish_nuclear.parquet")

new_nuclear_capacity_factor = 0.90  # Assumed capacity factor for new nuclear profile

years = [i for i in range(2015, 2025)]
installed_cap = [9648, 9102, 8629, 8613, 7740, 6882, 6882, 6937, 6944, 7008]
df_installed = pd.DataFrame(data=installed_cap, index=years, columns=["MW"])

df["day"] = df.index.day_of_year
year_start = pd.to_datetime(df.index.year.astype(str) + "-01-01").tz_localize(df.index.tz)
df["hour_of_year"] = ((df.index - year_start).total_seconds() // 3600).astype(int)
df["year"] = df.index.year

# Map installed capacity into the main df
df["installed_capacity"] = df["year"].map(df_installed["MW"])

# Calculate capacity factor
df["capacity_factor"] = df["Nuclear"] / df["installed_capacity"]


# --- Plot historic hourly generation ---
fig_hist = px.line(
    df.reset_index(),
    x="hour_of_year",
    y="capacity_factor",
    color="year",
    title="Swedish Nuclear Generation (Hourly)",
    labels={"index": "Timestamp", "Nuclear": "Generation (MW)"},
)
fig_hist.show()


# --- Profile ----

df_profile = df.groupby("hour_of_year")[["capacity_factor"]].mean()
df_profile = df_profile.reset_index()

df_profile["rolling"] = df_profile.rolling(window=24 * 14, min_periods=0).mean()["capacity_factor"]

fig_hist.add_trace(go.Line(x=df_profile["hour_of_year"], y=df_profile["capacity_factor"]))

fig_hist.add_trace(go.Line(x=df_profile["hour_of_year"], y=df_profile["rolling"]))

fig_hist.show()

df_profile["capacity_factor"].mean()

simulation_start = pd.Timestamp(year=2024, month=1, day=1, hour=0, minute=0, second=0, tz=CET_winter)
simulation_years = 1
simulation_end = simulation_start + pd.Timedelta(weeks=52 * simulation_years)

simulation_time_index = pd.date_range(start=simulation_start, end=simulation_end, freq="1h")

df_historic_profile = pd.DataFrame(
    data=df_profile["rolling"].iloc[: len(simulation_time_index)].values,
    index=simulation_time_index,
    columns=["capacity_factor"],
)

# save historic (capacity factor) profile
df_historic_profile.to_parquet("data/historic_nuclear_profile.parquet")
# df_historic_profile = pd.read_parquet("data/historic_nuclear_profile.parquet")

# Create a new profile scaled to the assumed new nuclear capacity factor
df_new_profile = df_historic_profile.copy()
for _ in range(20):
    df_new_profile = df_new_profile * new_nuclear_capacity_factor / df_new_profile.mean()
    df_new_profile = df_new_profile.clip(upper=1.0)

df_new_profile.to_parquet("data/new_nuclear_profile.parquet")
# df_new_profile = pd.read_parquet("data/new_nuclear_profile.parquet")


# Add month column to both profiles
df_historic_profile_month = df_historic_profile.copy()
df_historic_profile_month["month"] = df_historic_profile_month.index.month
df_historic_profile_month["day"] = df_historic_profile_month.index.day
df_historic_profile_month["hour"] = df_historic_profile_month.index.hour

df_new_profile_month = df_new_profile.copy()
df_new_profile_month["month"] = df_new_profile_month.index.month
df_new_profile_month["day"] = df_new_profile_month.index.day
df_new_profile_month["hour"] = df_new_profile_month.index.hour

# Group by month and calculate mean for each profile
historic_monthly = df_historic_profile_month.groupby("month")["capacity_factor"].mean()
new_monthly = df_new_profile_month.groupby("month")["capacity_factor"].mean()

fig_profiles = go.Figure()
fig_profiles.add_trace(
    go.Scatter(
        x=df_historic_profile_month.index,
        y=df_historic_profile_month["capacity_factor"],
        mode="markers",
        name="Historic nuclear (hourly)",
        marker=dict(color="steelblue", size=3, opacity=0.3),
        showlegend=True,
    )
)
fig_profiles.add_trace(
    go.Scatter(
        x=df_new_profile_month.index,
        y=df_new_profile_month["capacity_factor"],
        mode="markers",
        name=f"New nuclear (hourly, CF={new_nuclear_capacity_factor:.2f})",
        marker=dict(color="firebrick", size=3, opacity=0.3),
        showlegend=True,
    )
)

# Set x-axis ticks to months 1-12
month_ticks = [df_historic_profile_month.index[df_historic_profile_month["month"] == m][0] for m in range(1, 13)]
fig_profiles.update_xaxes(
    title="Month",
    tickmode="array",
    tickvals=month_ticks,
    ticktext=[str(m) for m in range(1, 13)],
    range=[df_historic_profile_month.index[0], df_historic_profile_month.index[-1]],
)
fig_profiles.update_yaxes(title="Capacity factor (-)", range=[0.60, 1.1])
fig_profiles = my_layout(fig_profiles, xaxis_title="Month", yaxis_title="Capacity factor (-)")
fig_profiles.update_layout(
    title="Historic vs. New Nuclear Capacity Factor Profiles by Month",
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black",
        borderwidth=1,
        yanchor="top",
        xanchor="left",
        font=dict(size=12),
    ),
)
fig_profiles.show()

fig_profiles.write_image("images/historic_vs_new_nuclear_capacity_factor_profiles.pdf")
