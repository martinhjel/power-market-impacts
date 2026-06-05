from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.processed_results import processed_data_path

DEFAULT_MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load_imp_nuke"
DEFAULT_REFERENCE_PRICE_EUR_MWH = 1000.0

NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
NORDIC_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4", "DK1", "DK2", "FI"]
OFFSHORE_WIND_AREAS = ["SNII", "UN", "VVD"]

REGION_PRICE_AREAS = {
    "NO": NO_AREAS,
    "NORDIC": NORDIC_AREAS,
}
REGION_CONSUMER_AREAS = {
    "NO": NO_AREAS,
    "NORDIC": NORDIC_AREAS,
}
REGION_PRODUCER_AREAS = {
    # Offshore wind busbars are represented separately in processed_data and are counted once here.
    "NO": NO_AREAS + OFFSHORE_WIND_AREAS,
    "NORDIC": NORDIC_AREAS + OFFSHORE_WIND_AREAS,
}

TECH_GENERATION_COLUMNS = {
    "hydro": "hydro",
    "solar": "solar",
    "wind_onshore": "onshore_wind",
    "wind_offshore": "offshore_wind",
    "nuclear": "_nuclear_generation",
    "biomass": "biomass",
    "fossil_gas": "fossil_gas",
    "fossil_other": "fossil_other",
}

TECH_COST_ALIASES = {
    "hydro": ["hydro"],
    "solar": ["solar"],
    "wind_onshore": ["wind_onshore", "onshore_wind", "wind onshore"],
    "wind_offshore": ["wind_offshore", "offshore_wind", "wind offshore"],
    "nuclear": ["nuclear", "nuclear (new)"],
    "biomass": ["biomass"],
    "fossil_gas": ["fossil_gas", "fossil gas"],
    "fossil_other": ["fossil_other", "fossil other"],
}

SCENARIO_COLORS = {
    "B23-BA": "#8c8c8c",
    "B23-LLPS": "#bdbdbd",
    "N-LLPS": "#1f77b4",
    "OWN-LLPS": "#2ca02c",
    "OW-LLPS": "#17becf",
    "N-BA": "#ff7f0e",
    "OWN-BA": "#d62728",
    "OW-BA": "#ff9896",
}
SURPLUS_COLORS = {
    "consumer": "#4c78a8",
    "producer": "#f58518",
    "societal": "#222222",
}
BUSBAR_AREAS = sorted(set(REGION_PRODUCER_AREAS["NORDIC"]))
BUSBAR_READ_COLUMNS = [
    "area",
    "weather_year",
    "price",
    "load",
    "hydro",
    "solar",
    "onshore_wind",
    "offshore_wind",
    "fixed_nuclear",
    "historic_nuclear",
    "new_nuclear",
    "total_nuclear",
    "biomass",
    "fossil_gas",
    "fossil_other",
]
REQUIRED_MARKET_STEP_COLUMNS = ["biomass", "fossil_gas", "fossil_other"]


@dataclass(frozen=True)
class ScenarioSpec:
    label: str
    scenario_name: str
    load_mode: str
    case: str
    hydro_uprated: bool
    is_baseline: bool = False


SCENARIOS = [
    ScenarioSpec(
        label="B23-BA",
        scenario_name="BASELINE_23TWh_BA_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_NoneOFF",
        load_mode="BA",
        case="BASELINE_23TWh",
        hydro_uprated=False,
        is_baseline=True,
    ),
    ScenarioSpec(
        label="B23-LLPS",
        scenario_name="BASELINE_23TWh_LLPS_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_NoneOFF",
        load_mode="LLPS",
        case="BASELINE_23TWh",
        hydro_uprated=False,
        is_baseline=True,
    ),
    ScenarioSpec(
        label="N-BA",
        scenario_name="BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        load_mode="BA",
        case="N",
        hydro_uprated=False,
    ),
    ScenarioSpec(
        label="OWN-BA",
        scenario_name="BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
        load_mode="BA",
        case="OWN",
        hydro_uprated=False,
    ),
    ScenarioSpec(
        label="OW-BA",
        scenario_name="BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
        load_mode="BA",
        case="OW",
        hydro_uprated=False,
    ),
    ScenarioSpec(
        label="N-LLPS",
        scenario_name="LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        load_mode="LLPS",
        case="N",
        hydro_uprated=False,
    ),
    ScenarioSpec(
        label="OWN-LLPS",
        scenario_name="LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
        load_mode="LLPS",
        case="OWN",
        hydro_uprated=False,
    ),
    ScenarioSpec(
        label="OW-LLPS",
        scenario_name="LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
        load_mode="LLPS",
        case="OW",
        hydro_uprated=False,
    ),
    ScenarioSpec(
        label="N-BA+",
        scenario_name="BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        load_mode="BA",
        case="N",
        hydro_uprated=True,
    ),
    ScenarioSpec(
        label="OWN-BA+",
        scenario_name="BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
        load_mode="BA",
        case="OWN",
        hydro_uprated=True,
    ),
    ScenarioSpec(
        label="OW-BA+",
        scenario_name="BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
        load_mode="BA",
        case="OW",
        hydro_uprated=True,
    ),
    ScenarioSpec(
        label="N-LLPS+",
        scenario_name="LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        load_mode="LLPS",
        case="N",
        hydro_uprated=True,
    ),
    ScenarioSpec(
        label="OWN-LLPS+",
        scenario_name="LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
        load_mode="LLPS",
        case="OWN",
        hydro_uprated=True,
    ),
    ScenarioSpec(
        label="OW-LLPS+",
        scenario_name="LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
        load_mode="LLPS",
        case="OW",
        hydro_uprated=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BASELINE_23TWh BA/LLPS processed results with N, OWN, and OW scenarios. "
            "Outputs CSV tables with mean prices and surplus deltas."
        )
    )
    parser.add_argument("--model-folder", default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--output-root", default="visualizations")
    parser.add_argument("--reference-price", type=float, default=DEFAULT_REFERENCE_PRICE_EUR_MWH)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any configured scenario is missing processed_data.parquet.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV and LaTeX outputs.")
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger("compare_23twh_scenarios")


def load_operational_costs(project_root: Path) -> dict[str, float]:
    costs_path = project_root / "data/operational_costs.json"
    if not costs_path.exists():
        costs_path = project_root / "data/operational_cost.json"
    if not costs_path.exists():
        raise FileNotFoundError("Missing data/operational_costs.json or data/operational_cost.json")

    raw = pd.read_json(costs_path, typ="series")
    costs: dict[str, float] = {}
    if all(not isinstance(value, (dict, list)) for value in raw.values):
        for key, value in raw.items():
            costs[str(key).strip().lower()] = float(value)
        return costs

    payload = pd.json_normalize(raw["technologies"])
    for _, row in payload.iterrows():
        costs[str(row["technology"]).strip().lower()] = float(row["operational_cost"])
    return costs


def operational_cost(costs: dict[str, float], tech: str) -> float:
    for candidate in TECH_COST_ALIASES.get(tech, [tech]):
        if candidate in costs:
            return costs[candidate]
    available = ", ".join(sorted(costs))
    raise KeyError(f"Missing operational cost for {tech}. Available keys: {available}")


def existing_specs(project_root: Path, model_folder: str, strict: bool, logger: logging.Logger) -> list[ScenarioSpec]:
    available = []
    missing = []
    for spec in SCENARIOS:
        path = processed_data_path(project_root, model_folder, spec.scenario_name)
        if path.exists():
            available.append(spec)
        else:
            missing.append(spec)

    if missing:
        message = "Missing processed results: " + ", ".join(spec.label for spec in missing)
        if strict:
            raise FileNotFoundError(message)
        logger.warning(message)
    return available


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype("float64", copy=False)


def read_processed_busbar_frame(project_root: Path, model_folder: str, spec: ScenarioSpec) -> pd.DataFrame:
    data_path = processed_data_path(project_root, model_folder, spec.scenario_name)
    schema_columns = set(pq.ParquetFile(data_path).schema_arrow.names)
    missing_market_step_columns = [column for column in REQUIRED_MARKET_STEP_COLUMNS if column not in schema_columns]
    if missing_market_step_columns:
        raise RuntimeError(
            f"{data_path} is missing {missing_market_step_columns}. "
            "Re-run scripts/process_ltm_results.py with --overwrite."
        )

    columns = [column for column in BUSBAR_READ_COLUMNS if column in schema_columns]
    df = pd.read_parquet(
        data_path,
        columns=columns,
        filters=[("record_type", "==", "busbar"), ("area", "in", BUSBAR_AREAS)],
    )
    if df.empty:
        raise RuntimeError(f"No busbar rows found in {data_path}")
    return df


def nuclear_generation(df: pd.DataFrame) -> pd.Series:
    total = numeric_series(df, "total_nuclear", default=np.nan)
    fallback_columns = [column for column in ("historic_nuclear", "new_nuclear") if column in df.columns]
    if fallback_columns:
        fallback = sum(numeric_series(df, column) for column in fallback_columns)
    else:
        fallback = numeric_series(df, "fixed_nuclear")
    return total.fillna(fallback).fillna(0.0)


def calculate_area_metrics_from_processed_frame(
    *,
    df: pd.DataFrame,
    costs: dict[str, float],
    reference_price: float,
) -> dict[str, dict[str, float]]:
    n_weather_years = max(1, int(df["weather_year"].nunique())) if "weather_year" in df.columns else 1
    work = df[["area"]].copy()
    work["_price"] = numeric_series(df, "price")
    work["_load"] = numeric_series(df, "load")
    work["_price_load_sum"] = work["_price"] * work["_load"]
    work["_price_sq_load_sum"] = work["_price"] ** 2 * work["_load"]
    work["_consumer_surplus_eur"] = work["_load"] * (reference_price - work["_price"])
    work["_generation_mwh"] = 0.0
    work["_producer_surplus_eur"] = 0.0

    for tech, column in TECH_GENERATION_COLUMNS.items():
        generation = nuclear_generation(df) if column == "_nuclear_generation" else numeric_series(df, column)
        work["_generation_mwh"] += generation
        work["_producer_surplus_eur"] += generation * (work["_price"] - operational_cost(costs, tech))

    grouped = work.groupby("area", observed=True, sort=True).agg(
        mean_price_eur_mwh=("_price", "mean"),
        price_load_sum=("_price_load_sum", "sum"),
        price_sq_load_sum=("_price_sq_load_sum", "sum"),
        load_mwh=("_load", "sum"),
        generation_mwh=("_generation_mwh", "sum"),
        consumer_surplus_eur=("_consumer_surplus_eur", "sum"),
        producer_surplus_eur=("_producer_surplus_eur", "sum"),
    )
    grouped["load_weighted_price_eur_mwh"] = grouped["price_load_sum"] / grouped["load_mwh"].replace(0.0, np.nan)
    grouped["load_weighted_price_eur_mwh"] = grouped["load_weighted_price_eur_mwh"].fillna(
        grouped["mean_price_eur_mwh"]
    )
    grouped["price_std_eur_mwh"] = np.sqrt(
        (
            grouped["price_sq_load_sum"] / grouped["load_mwh"].replace(0.0, np.nan)
            - grouped["load_weighted_price_eur_mwh"] ** 2
        ).clip(lower=0.0)
    )
    grouped["price_load_sum"] = grouped["price_load_sum"] / n_weather_years
    grouped["price_sq_load_sum"] = grouped["price_sq_load_sum"] / n_weather_years
    grouped["load_twh"] = grouped["load_mwh"] / n_weather_years / 1e6
    grouped["generation_twh"] = grouped["generation_mwh"] / n_weather_years / 1e6
    grouped["consumer_surplus_meur"] = grouped["consumer_surplus_eur"] / n_weather_years / 1e6
    grouped["producer_surplus_meur"] = grouped["producer_surplus_eur"] / n_weather_years / 1e6
    grouped["societal_surplus_meur"] = grouped["consumer_surplus_meur"] + grouped["producer_surplus_meur"]

    metric_columns = [
        "mean_price_eur_mwh",
        "load_weighted_price_eur_mwh",
        "price_std_eur_mwh",
        "price_load_sum",
        "price_sq_load_sum",
        "load_twh",
        "generation_twh",
        "consumer_surplus_meur",
        "producer_surplus_meur",
        "societal_surplus_meur",
    ]
    return grouped[metric_columns].to_dict(orient="index")


def aggregate_metrics(
    area_results: dict[str, dict[str, float]],
    *,
    price_areas: Iterable[str],
    consumer_areas: Iterable[str],
    producer_areas: Iterable[str],
) -> dict[str, float]:
    price_load_sum = sum(area_results[area]["price_load_sum"] for area in price_areas if area in area_results)
    price_sq_load_sum = sum(
        area_results[area]["price_sq_load_sum"] for area in price_areas if area in area_results
    )
    load_twh = sum(area_results[area]["load_twh"] for area in consumer_areas if area in area_results)
    load_mwh_annual = load_twh * 1e6
    mean_price = price_load_sum / load_mwh_annual if load_mwh_annual > 0 else np.nan
    price_variance = price_sq_load_sum / load_mwh_annual - mean_price**2 if load_mwh_annual > 0 else np.nan
    price_std = np.sqrt(max(price_variance, 0.0)) if np.isfinite(price_variance) else np.nan

    consumer_surplus = sum(
        area_results[area]["consumer_surplus_meur"] for area in consumer_areas if area in area_results
    )
    producer_surplus = sum(
        area_results[area]["producer_surplus_meur"] for area in producer_areas if area in area_results
    )
    generation_twh = sum(area_results[area]["generation_twh"] for area in producer_areas if area in area_results)

    return {
        "mean_price_eur_mwh": mean_price,
        "price_std_eur_mwh": price_std,
        "load_twh": load_twh,
        "generation_twh": generation_twh,
        "consumer_surplus_meur": consumer_surplus,
        "producer_surplus_meur": producer_surplus,
        "societal_surplus_meur": consumer_surplus + producer_surplus,
    }


def calculate_scenario_metrics(
    spec: ScenarioSpec,
    project_root: Path,
    model_folder: str,
    costs: dict[str, float],
    reference_price: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    df = read_processed_busbar_frame(project_root, model_folder, spec)
    by_area = calculate_area_metrics_from_processed_frame(
        df=df,
        costs=costs,
        reference_price=reference_price,
    )

    by_region = {}
    for region in REGION_PRICE_AREAS:
        by_region[region] = aggregate_metrics(
            by_area,
            price_areas=REGION_PRICE_AREAS[region],
            consumer_areas=REGION_CONSUMER_AREAS[region],
            producer_areas=REGION_PRODUCER_AREAS[region],
        )

    for metrics in [*by_area.values(), *by_region.values()]:
        metrics["scenario"] = spec.label
        metrics["scenario_folder"] = spec.scenario_name
        metrics["load_mode"] = spec.load_mode
        metrics["case"] = spec.case
        metrics["hydro_uprated"] = spec.hydro_uprated
        metrics["is_baseline"] = spec.is_baseline

    return by_area, by_region


def add_deltas(df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    out = df.copy()
    metric_columns = [
        "mean_price_eur_mwh",
        "price_std_eur_mwh",
        "load_twh",
        "generation_twh",
        "consumer_surplus_meur",
        "producer_surplus_meur",
        "societal_surplus_meur",
    ]
    baseline_rows = out[out["is_baseline"]].set_index(["load_mode", key_column])
    out["baseline"] = pd.Series(pd.NA, index=out.index, dtype="object")

    for column in metric_columns:
        out[f"delta_{column}"] = np.nan
        out[f"delta_pct_{column}"] = np.nan

    for idx, row in out.iterrows():
        baseline_key = (row["load_mode"], row[key_column])
        if baseline_key not in baseline_rows.index:
            continue
        baseline = baseline_rows.loc[baseline_key]
        out.at[idx, "baseline"] = baseline["scenario"]
        for column in metric_columns:
            base_value = float(baseline[column])
            value = float(row[column])
            delta = value - base_value
            out.at[idx, f"delta_{column}"] = delta
            if abs(base_value) > 1e-12:
                out.at[idx, f"delta_pct_{column}"] = 100.0 * delta / base_value

    return out


def scenario_order_map() -> dict[str, int]:
    return {spec.label: i for i, spec in enumerate(SCENARIOS)}


def scenario_color(label: str) -> str:
    base_label = label.rstrip("+")
    return SCENARIO_COLORS.get(label, SCENARIO_COLORS.get(base_label, "#7f7f7f"))


def scenario_hatch(label: str) -> str | None:
    return "//" if label.endswith("+") else None


def ordered_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario_order"] = out["scenario"].map(scenario_order_map())
    return out.sort_values("scenario_order").drop(columns=["scenario_order"])


def apply_common_axis_style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_regional_metric_comparison(
    *,
    region_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    metric_column: str,
    y_label: str,
    title: str,
    output_filename: str,
    sharey: bool = True,
    tight_y_axis: bool = False,
    value_scale: float = 1.0,
    include_baselines: bool = True,
    zero_line: bool = False,
) -> None:
    plot_df = region_df.dropna(subset=[metric_column])
    if not include_baselines:
        plot_df = plot_df[~plot_df["is_baseline"]]
    regions = [region for region in ["NO", "NORDIC"] if region in set(plot_df["region"])]
    if not regions:
        logger.warning("Skipping %s because no regional data is available", output_filename)
        return

    fig, axes = plt.subplots(1, len(regions), figsize=(12, 4.2), sharey=sharey)
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        sub = ordered_rows(plot_df[plot_df["region"] == region])
        x = np.arange(len(sub))
        if zero_line:
            ax.axhline(0, color="#333333", linewidth=0.8)
        bars = ax.bar(
            x,
            sub[metric_column] * value_scale,
            color=[scenario_color(label) for label in sub["scenario"]],
            edgecolor="#333333",
            linewidth=0.5,
        )
        for bar, label in zip(bars, sub["scenario"]):
            hatch = scenario_hatch(label)
            if hatch:
                bar.set_hatch(hatch)
        ax.set_title(region)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["scenario"], rotation=45, ha="right")
        ax.set_ylabel(y_label)
        if tight_y_axis:
            values = (sub[metric_column] * value_scale).to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                min_value = float(np.min(finite_values))
                max_value = float(np.max(finite_values))
                if zero_line:
                    min_value = min(min_value, 0.0)
                    max_value = max(max_value, 0.0)
                padding = max((max_value - min_value) * 0.08, abs(max_value) * 0.005, 1.0)
                ax.set_ylim(min_value - padding, max_value + padding)
        apply_common_axis_style(ax)

    fig.suptitle(title)
    fig.tight_layout()
    output_path = output_dir / output_filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def plot_mean_price_comparison(region_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    plot_regional_metric_comparison(
        region_df=region_df,
        output_dir=output_dir,
        logger=logger,
        metric_column="mean_price_eur_mwh",
        y_label="Mean price (EUR/MWh)",
        title="Mean Power Price Compared with 23 TWh Baselines",
        output_filename="baseline_23twh_mean_price_comparison.pdf",
        sharey=False,
        tight_y_axis=True,
    )


def plot_price_std_comparison(region_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    plot_regional_metric_comparison(
        region_df=region_df,
        output_dir=output_dir,
        logger=logger,
        metric_column="price_std_eur_mwh",
        y_label="Price standard deviation (EUR/MWh)",
        title="Expected Power Price Standard Deviation Compared with 23 TWh Baselines",
        output_filename="baseline_23twh_price_std_comparison.pdf",
        sharey=False,
        tight_y_axis=True,
    )


def plot_consumer_surplus_comparison(region_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    plot_regional_metric_comparison(
        region_df=region_df,
        output_dir=output_dir,
        logger=logger,
        metric_column="delta_consumer_surplus_meur",
        y_label="Change in consumer surplus (BEUR/year)",
        title="Consumer Surplus Change from 23 TWh Baselines",
        output_filename="baseline_23twh_consumer_surplus_comparison.pdf",
        sharey=False,
        tight_y_axis=True,
        value_scale=1 / 1000,
        include_baselines=False,
        zero_line=True,
    )


def plot_producer_surplus_comparison(region_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    plot_regional_metric_comparison(
        region_df=region_df,
        output_dir=output_dir,
        logger=logger,
        metric_column="producer_surplus_meur",
        y_label="Producer surplus (MEUR/year)",
        title="Producer Surplus Compared with 23 TWh Baselines",
        output_filename="baseline_23twh_producer_surplus_comparison.pdf",
    )


def plot_surplus_delta_comparison(region_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    delta_columns = ["delta_consumer_surplus_meur", "delta_producer_surplus_meur", "delta_societal_surplus_meur"]
    plot_df = region_df[(~region_df["is_baseline"]) & region_df[delta_columns].notna().all(axis=1)]
    regions = [region for region in ["NO", "NORDIC"] if region in set(plot_df["region"])]
    if not regions:
        logger.warning("Skipping surplus delta plot because no baseline deltas are available")
        return

    fig, axes = plt.subplots(1, len(regions), figsize=(13, 4.5), sharey=True)
    if len(regions) == 1:
        axes = [axes]

    width = 0.36
    for ax, region in zip(axes, regions):
        sub = ordered_rows(plot_df[plot_df["region"] == region])
        x = np.arange(len(sub))
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.bar(
            x - width / 2,
            sub["delta_consumer_surplus_meur"],
            width=width,
            color=SURPLUS_COLORS["consumer"],
            label="Consumer" if ax is axes[0] else None,
        )
        ax.bar(
            x + width / 2,
            sub["delta_producer_surplus_meur"],
            width=width,
            color=SURPLUS_COLORS["producer"],
            label="Producer" if ax is axes[0] else None,
        )
        ax.plot(
            x,
            sub["delta_societal_surplus_meur"],
            color=SURPLUS_COLORS["societal"],
            marker="o",
            linestyle="none",
            markersize=4,
            label="Total" if ax is axes[0] else None,
        )
        ax.set_title(region)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["scenario"], rotation=45, ha="right")
        ax.set_ylabel("Change from 23 TWh baseline (MEUR/year)")
        apply_common_axis_style(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Producer and Consumer Surplus Change")
    fig.tight_layout()
    output_path = output_dir / "baseline_23twh_surplus_delta_comparison.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def plot_no_area_price_delta_heatmap(area_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    plot_df = area_df[
        area_df["area"].isin(NO_AREAS)
        & (~area_df["is_baseline"])
        & area_df["delta_mean_price_eur_mwh"].notna()
    ]
    if plot_df.empty:
        logger.warning("Skipping NO area price delta heatmap because no baseline deltas are available")
        return

    ordered = ordered_rows(plot_df)
    matrix = ordered.pivot_table(
        index="scenario",
        columns="area",
        values="delta_mean_price_eur_mwh",
        aggfunc="first",
    ).reindex(columns=NO_AREAS)
    matrix = matrix.reindex([label for label in scenario_order_map() if label in matrix.index])

    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        logger.warning("Skipping NO area price delta heatmap because all deltas are NaN")
        return

    bound = max(abs(float(np.nanmin(finite_values))), abs(float(np.nanmax(finite_values))), 1.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
    )
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("NO Area Mean Price Change from 23 TWh Baseline")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                text_color = "white" if abs(value) > bound * 0.55 else "#222222"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("EUR/MWh")
    fig.tight_layout()
    output_path = output_dir / "baseline_23twh_no_area_price_delta_heatmap.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def write_plots(output_dir: Path, region_df: pd.DataFrame, area_df: pd.DataFrame, logger: logging.Logger) -> None:
    plot_mean_price_comparison(region_df, output_dir, logger)
    plot_price_std_comparison(region_df, output_dir, logger)
    plot_consumer_surplus_comparison(region_df, output_dir, logger)
    plot_producer_surplus_comparison(region_df, output_dir, logger)
    plot_surplus_delta_comparison(region_df, output_dir, logger)
    plot_no_area_price_delta_heatmap(area_df, output_dir, logger)


def write_outputs(
    *,
    output_dir: Path,
    region_df: pd.DataFrame,
    area_df: pd.DataFrame,
    logger: logging.Logger,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    region_path = output_dir / "baseline_23twh_scenario_comparison.csv"
    area_path = output_dir / "baseline_23twh_scenario_comparison_by_area.csv"
    tex_path = output_dir / "baseline_23twh_scenario_comparison.tex"

    region_df.to_csv(region_path, index=False)
    area_df.to_csv(area_path, index=False)

    table_columns = [
        "scenario",
        "region",
        "baseline",
        "mean_price_eur_mwh",
        "delta_mean_price_eur_mwh",
        "consumer_surplus_meur",
        "delta_consumer_surplus_meur",
        "producer_surplus_meur",
        "delta_producer_surplus_meur",
        "societal_surplus_meur",
        "delta_societal_surplus_meur",
    ]
    table = region_df[table_columns].copy()
    numeric_columns = [column for column in table.columns if column not in {"scenario", "region", "baseline"}]
    for column in numeric_columns:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.1f}")
    table.to_latex(
        tex_path,
        index=False,
        escape=True,
        caption=(
            "Comparison of 23 TWh baseline scenarios with N, OWN, and OW scenarios. "
            "Prices are load-weighted regional means. Surplus values are annual expected MEUR, "
            "and deltas are relative to the 23 TWh baseline with the same load allocation method."
        ),
        label="tab:baseline_23twh_scenario_comparison",
    )

    logger.info("Wrote %s", region_path)
    logger.info("Wrote %s", area_path)
    logger.info("Wrote %s", tex_path)

    if make_plots:
        write_plots(output_dir, region_df, area_df, logger)


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    project_root = PROJECT_ROOT
    output_dir = project_root / args.output_root / args.model_folder / "paper"
    costs = load_operational_costs(project_root)
    specs = existing_specs(project_root, args.model_folder, args.strict, logger)

    if not specs:
        raise SystemExit("No configured processed results were found.")

    area_rows = []
    region_rows = []
    for spec in specs:
        logger.info("Loading %s", spec.label)
        by_area, by_region = calculate_scenario_metrics(
            spec,
            project_root=project_root,
            model_folder=args.model_folder,
            costs=costs,
            reference_price=args.reference_price,
        )
        for area, metrics in by_area.items():
            area_rows.append({"area": area, **metrics})
        for region, metrics in by_region.items():
            region_rows.append({"region": region, **metrics})

    if not region_rows:
        raise SystemExit("No metrics could be calculated.")

    region_df = pd.DataFrame(region_rows)
    area_df = pd.DataFrame(area_rows)

    region_df = add_deltas(region_df, key_column="region")
    area_df = add_deltas(area_df, key_column="area")

    ordering = scenario_order_map()
    region_order = {"NO": 0, "NORDIC": 1}
    region_df["scenario_order"] = region_df["scenario"].map(ordering)
    region_df["region_order"] = region_df["region"].map(region_order)
    region_df = region_df.sort_values(["region_order", "scenario_order"]).drop(
        columns=["scenario_order", "region_order"]
    )

    area_df["scenario_order"] = area_df["scenario"].map(ordering)
    area_df = area_df.sort_values(["area", "scenario_order"]).drop(columns=["scenario_order"])

    write_outputs(
        output_dir=output_dir,
        region_df=region_df,
        area_df=area_df,
        logger=logger,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
