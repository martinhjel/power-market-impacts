from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")
st.title("📊 Market Dispatch")

RESULT_ROOT = Path.cwd() / "ltm_output"


def _result_label(p: Path) -> str:
    parent = p.parent.name
    if "1H" in parent:
        res = "1H"
    elif "1D" in parent:
        res = "1D"
    else:
        res = parent
    return f"{p.name} ({res})"


result_paths = sorted([i for i in RESULT_ROOT.glob("*/*") if i.is_dir()], key=lambda p: p.name.lower())

selected_path = st.sidebar.selectbox(
    "Select simulation:",
    result_paths,
    format_func=_result_label,
    index=0,
)

st.sidebar.markdown(f"**Path:** `{selected_path}`")

if st.sidebar.button("Reload Data"):
    st.cache_data.clear()
    st.success("Cache cleared. Data will reload.")

processed_dir = selected_path / "results" / "processed"
export_file = processed_dir / "market_dispatch.pkl"
process_file = processed_dir / "market_dispatch.parquet"

if not processed_dir.exists():
    st.error(f"Processed results directory not found: {processed_dir}")
    st.stop()


@st.cache_data
def _read_export(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


@st.cache_data
def _read_process(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _index_values(df: pd.DataFrame, level: str) -> List:
    return list(df.index.get_level_values(level).unique())


def _default_choice(options: Iterable, preferred) -> int:
    try:
        return list(options).index(preferred)
    except ValueError:
        return 0


def _plot_timeseries(df: pd.DataFrame, title: str, columns: List[str]) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df.index.get_level_values("timestamp"),
                y=df[col],
                name=col,
                mode="lines",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="MW",
        legend_title="Series",
        height=500,
    )
    return fig


if not export_file.exists() and not process_file.exists():
    st.error("No market dispatch outputs found in results/processed.")
    st.info("Run market_dispatch_export.py and market_dispatch_process.py first.")
    st.stop()

export_df = None
process_df = None

if export_file.exists():
    export_df = _read_export(export_file)
else:
    st.warning(f"Missing export file: {export_file}")

if process_file.exists():
    process_df = _read_process(process_file)
else:
    st.warning(f"Missing process file: {process_file}")

areas = []
scenarios = []

if export_df is not None:
    areas = _index_values(export_df, "area")
    scenarios = _index_values(export_df, "scenario")
if process_df is not None:
    areas = sorted(set(areas) | set(_index_values(process_df, "area")))
    scenarios = sorted(set(scenarios) | set(_index_values(process_df, "scenario")))

if not areas or not scenarios:
    st.error("No areas or scenarios found in the processed files.")
    st.stop()

area_default = _default_choice(areas, "NO2")
scenario_default = _default_choice(scenarios, 0)

col_a, col_s = st.columns(2)
with col_a:
    area = st.selectbox("Area", areas, index=area_default)
with col_s:
    scenario = st.selectbox("Scenario", scenarios, index=scenario_default)

left, right = st.columns(2)

with left:
    st.subheader("Market Dispatch Export")
    if export_df is None:
        st.info("No export data available.")
    else:
        df_exp = export_df.loc[(area, scenario)]
        exp_cols = [
            "load",
            "hydro",
            "onshore_wind",
            "offshore_wind",
            "solar",
            "market_steps",
            "net_import_export",
        ]
        fig = _plot_timeseries(df_exp, f"Export: {area} | Scenario {scenario}", exp_cols)
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Market Dispatch Process")
    if process_df is None:
        st.info("No process data available.")
    else:
        df_proc = process_df.loc[(area, scenario)]
        core_cols = ["total", "sum_market_steps"]
        drop_cols = {"market_price", "diff"}
        proc_cols = core_cols + [c for c in df_proc.columns if c not in drop_cols and c not in core_cols]
        fig = _plot_timeseries(df_proc, f"Process: {area} | Scenario {scenario}", proc_cols)
        st.plotly_chart(fig, use_container_width=True)

if process_df is not None:
    df_proc = process_df.loc[(area, scenario)]
    if "market_price" in df_proc.columns:
        st.subheader("Market Price")
        price_fig = go.Figure(
            [
                go.Scatter(
                    x=df_proc.index.get_level_values("timestamp"),
                    y=df_proc["market_price"],
                    name="market_price",
                    mode="lines",
                )
            ]
        )
        price_fig.update_layout(
            title=f"Market Price: {area} | Scenario {scenario}",
            xaxis_title="Timestamp",
            yaxis_title="€/MWh",
            height=400,
        )
        st.plotly_chart(price_fig, use_container_width=True)
