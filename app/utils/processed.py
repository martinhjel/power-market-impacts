from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from scripts.processed_results import ProcessedScenarioResults

PROCESSED_ROOT = Path.cwd() / "ltm_processed"


def list_processed_scenarios(root: Path = PROCESSED_ROOT) -> list[Path]:
    return sorted(
        [path.parent for path in root.glob("*/*/processed_data.parquet")],
        key=lambda path: (path.parent.name.lower(), path.name.lower()),
    )


def result_label(path: Path) -> str:
    try:
        model, scenario = path.relative_to(PROCESSED_ROOT).parts[:2]
    except ValueError:
        model, scenario = path.parent.name, path.name

    tags = []
    if "1H" in model:
        tags.append("1H")
    elif "1D" in model:
        tags.append("1D")
    if "imp_nuke" in model:
        tags.append("imp nuke")

    suffix = f" ({', '.join(tags)})" if tags else ""
    return f"{model}/{scenario}{suffix}"


@st.cache_resource(show_spinner="Loading processed scenario...")
def load_processed_scenario(path: str) -> ProcessedScenarioResults:
    scenario_path = Path(path)
    data_path = scenario_path / "processed_data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed_data.parquet in {scenario_path}")
    return ProcessedScenarioResults(data_path=data_path)


@st.cache_data(show_spinner=False)
def busbar_names(path: str) -> list[str]:
    return load_processed_scenario(path).get_busbar_names()


@st.cache_data(show_spinner=False)
def dcline_names(path: str) -> list[str]:
    return load_processed_scenario(path).get_dcline_names()


@st.cache_data(show_spinner=False)
def busbar_metric(path: str, area: str, metric: str) -> pd.DataFrame:
    return load_processed_scenario(path).get_busbar_metric(area, metric)


@st.cache_data(show_spinner=False)
def dcline_flow(path: str, dcline_name: str) -> pd.DataFrame:
    return load_processed_scenario(path).get_dcline_flow(dcline_name)


def safe_busbar_metric(path: str, area: str, metric: str, like: pd.DataFrame | None = None) -> pd.DataFrame:
    try:
        return busbar_metric(path, area, metric)
    except KeyError:
        if like is None:
            raise
        return pd.DataFrame(0.0, index=like.index, columns=like.columns)


def select_timeseries(df: pd.DataFrame, weather_year: int | str) -> pd.Series:
    if weather_year == "Mean":
        return df.mean(axis=1)

    try:
        return df[int(weather_year)]
    except KeyError:
        return df.iloc[:, int(weather_year)]


def default_index(options: Iterable, preferred) -> int:
    options = list(options)
    try:
        return options.index(preferred)
    except ValueError:
        return 0


def parse_dcline_endpoints(name: str) -> tuple[str, str] | None:
    clean = name.removeprefix("dcline_")
    if " " in clean:
        parts = clean.split()
    else:
        parts = clean.split("_")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def connected_dcline_flows(
    *,
    path: str,
    area: str,
    weather_year: int | str,
) -> dict[str, pd.Series]:
    flows: dict[str, pd.Series] = {}
    for dcline_name in dcline_names(path):
        endpoints = parse_dcline_endpoints(dcline_name)
        if endpoints is None:
            continue
        node_a, node_b = endpoints
        if area == node_a:
            sign = -1.0
            other = node_b
        elif area == node_b:
            sign = 1.0
            other = node_a
        else:
            continue
        flow = dcline_flow(path, dcline_name)
        flows[other] = sign * select_timeseries(flow, weather_year)
    return flows


def net_import_export(path: str, area: str, weather_year: int | str, index: pd.Index) -> pd.Series:
    total = pd.Series(0.0, index=index)
    for flow in connected_dcline_flows(path=path, area=area, weather_year=weather_year).values():
        total = total.add(flow, fill_value=0.0)
    return total
