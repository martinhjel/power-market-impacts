"""
Plot historical reservoir trajectory for configurable scenarios.
Supports per-scenario mean-only mode (no p10-p90 fill and no median line).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

from scripts.common import load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
NO_BUSBARS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

def _resolve_base_path() -> Path:
    p = Path.cwd()
    if p.name == "paper":
        return p.parent.parent
    if p.name == "scripts":
        return p.parent
    return p


def _load_historical_data(base_path: Path, num_steps: int) -> dict[str, np.ndarray] | None:
    historical_data_path = base_path / "app/data/historic_reservoir_nve.parquet"
    if not historical_data_path.exists():
        logger.warning(f"Historical data file not found at {historical_data_path}")
        return None

    try:
        df_hist = pd.read_parquet(historical_data_path)
        df_hist_norway = df_hist.loc[df_hist["omrType"] == "NO"]
        if df_hist_norway.empty:
            logger.warning("Historical data loaded but has no NO rows")
            return None

        df_hist_norway = df_hist_norway.set_index("dato_Id").sort_index()
        df_hist_norway["iso_uke"] = df_hist_norway.index.isocalendar().week

        weekly = df_hist_norway.groupby("iso_uke")["fyllingsgrad"].agg(
            [
                ("mean", "mean"),
                ("p10", lambda x: x.quantile(0.10)),
                ("p90", lambda x: x.quantile(0.90)),
            ]
        )

        hist_mean = np.tile(weekly["mean"].values * 100, (num_steps // 52) + 1)[:num_steps]
        hist_p10 = np.tile(weekly["p10"].values * 100, (num_steps // 52) + 1)[:num_steps]
        hist_p90 = np.tile(weekly["p90"].values * 100, (num_steps // 52) + 1)[:num_steps]
        logger.info("Loaded historical reservoir data from NVE")
        return {"mean": hist_mean, "p10": hist_p10, "p90": hist_p90}
    except Exception as e:
        logger.warning(f"Failed to load historical reservoir data: {e}")
        return None


def plot_reservoir_trajectory(
    scenarios_config: dict[str, str],
    mean_only_scenarios: list[str] | set[str] | tuple[str, ...] | dict[str, str] | None = None,
    output_name: str = "reservoir_trajectory.pdf",
    percentile_range: tuple[float, float] = (10.0, 90.0),
    include_historical: bool = True,
) -> plt.Figure:
    """
    Plot reservoir trajectory for selected scenarios.

    Args:
        scenarios_config: Mapping {scenario_folder_name: plot_label}.
        mean_only_scenarios:
            - list/set/tuple of selectors (folder name or plot label), or
            - dict {scenario_folder_name: plot_label} for additional mean-only scenarios.
        output_name: Output PDF filename in paper output folder.
        percentile_range: Lower/upper percentile for scenario range fill (e.g. (10, 90)).
        include_historical: Whether to load and plot historical trajectory/range.

    Returns:
        Matplotlib figure handle.
    """
    p_low, p_high = percentile_range
    if not (0 <= p_low < p_high <= 100):
        raise ValueError(f"Invalid percentile_range={percentile_range}. Expected 0 <= low < high <= 100.")

    base_path = _resolve_base_path()
    output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
    paper_output_path = output_path / "paper"
    paper_output_path.mkdir(parents=True, exist_ok=True)

    scenarios_to_plot = dict(scenarios_config)
    selectors: set[str] = set()

    if isinstance(mean_only_scenarios, dict):
        # Include additional scenarios and mark both folder names and labels as mean-only selectors.
        scenarios_to_plot.update(mean_only_scenarios)
        selectors.update(mean_only_scenarios.keys())
        selectors.update(mean_only_scenarios.values())
    elif mean_only_scenarios:
        selectors.update(mean_only_scenarios)

    scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in scenarios_to_plot.keys()}
    scenarios = load_scenarios(scenario_paths)

    if not scenarios:
        raise RuntimeError("No scenarios loaded")

    logger.info(f"Loaded {len(scenarios)} scenarios")
    logger.info(f"Configured scenarios: {scenarios_to_plot}")
    if selectors:
        logger.info(f"Mean-only selectors: {sorted(selectors)}")

    scenario_data: dict[str, dict[str, pd.Series]] = {}
    scenario_meta: dict[str, dict[str, object]] = {}

    for scenario_name, label in scenarios_to_plot.items():
        if scenario_name not in scenarios:
            logger.warning(f"Scenario {scenario_name} not found")
            continue

        scenario = scenarios[scenario_name]
        logger.info(f"Processing scenario: {label}")

        total_reservoir = None
        max_volume = 0.0

        for area in NO_BUSBARS:
            try:
                busbar = scenario.get_busbars()[area]
                for r in busbar.reservoirs():
                    val = df_from_pyltm_result(r.reservoir(time_axis=True))
                    total_reservoir = val if total_reservoir is None else total_reservoir + val
                    max_volume += val.max().max()
            except Exception as e:
                logger.warning(f"  Failed to get reservoir for {area}: {e}")

        if total_reservoir is None or max_volume == 0:
            logger.error(f"No reservoir data collected for {label}")
            continue

        total_pct = (total_reservoir / max_volume) * 100
        scenario_data[label] = {
            "mean": total_pct.mean(axis=1),
            "p_low": total_pct.quantile(p_low / 100.0, axis=1),
            "p_high": total_pct.quantile(p_high / 100.0, axis=1),
        }
        scenario_meta[label] = {
            "scenario_name": scenario_name,
            "mean_only": (label in selectors) or (scenario_name in selectors),
        }

    if not scenario_data:
        raise RuntimeError("No scenario data collected")

    time_steps = np.arange(len(next(iter(scenario_data.values()))["mean"]))
    historical_data = _load_historical_data(base_path, len(time_steps)) if include_historical else None

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Assign colors by base label so related scenarios (e.g. N-BA and N-BA+) share color.
    def _base_label(lbl: str) -> str:
        return lbl.replace("+", "").strip()

    base_order = []
    for lbl in scenario_data.keys():
        base = _base_label(lbl)
        if base not in base_order:
            base_order.append(base)
    cmap = plt.get_cmap("tab10")
    base_colors = {base: cmap(i % 10) for i, base in enumerate(base_order)}

    if historical_data:
        ax.fill_between(
            time_steps,
            historical_data["p10"],
            historical_data["p90"],
            alpha=0.2,
            color="gray",
            label="Historical p10-p90 range",
        )
        ax.plot(
            time_steps,
            historical_data["mean"],
            color="black",
            linewidth=2.0,
            label="Historical Mean",
            linestyle="-",
            alpha=0.7,
        )

    for label, data in scenario_data.items():
        color = base_colors[_base_label(label)]
        mean_only = bool(scenario_meta.get(label, {}).get("mean_only", False))
        linestyle = "--" if "+" in label else "-"

        if not mean_only:
            ax.fill_between(
                time_steps,
                data["p_low"],
                data["p_high"],
                alpha=0.2,
                color=color,
                label=f"{label} p{p_low:g}-p{p_high:g} range",
            )

        ax.plot(
            time_steps,
            data["mean"],
            color=color,
            linewidth=2.5,
            label=f"{label} Mean",
            linestyle=linestyle,
        )

    ax.set_xlabel("Time Step (week)", fontsize=12)
    ax.set_ylabel("Total Reservoir Filling (%)", fontsize=12)
    ax.set_title("Norway Total Reservoir Trajectory", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0, top=100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.95, fontsize=10, ncol=2)

    # Time steps are weekly in this workflow.
    # Map index to ISO-like week number 1..52.
    week_index = (time_steps % 52) + 1
    stats_text = ""
    for label, data in scenario_data.items():
        mean_values = np.asarray(data["mean"], dtype=float)
        w15 = float(np.nanmean(mean_values[week_index == 15])) if np.any(week_index == 15) else float("nan")
        w40 = float(np.nanmean(mean_values[week_index == 40])) if np.any(week_index == 40) else float("nan")
        stats_text += (
            f"{label}:\n"
            f"  Mean: {data['mean'].mean():.1f}%\n"
            f"  Week 15 mean: {w15:.1f}%\n"
            f"  Week 40 mean: {w40:.1f}%\n\n"
        )
    if historical_data:
        hist_mean_values = np.asarray(historical_data["mean"], dtype=float)
        hist_w15 = (
            float(np.nanmean(hist_mean_values[week_index == 15])) if np.any(week_index == 15) else float("nan")
        )
        hist_w40 = (
            float(np.nanmean(hist_mean_values[week_index == 40])) if np.any(week_index == 40) else float("nan")
        )
        stats_text += (
            "Historical:\n"
            f"  Mean: {np.mean(historical_data['mean']):.1f}%\n"
            f"  Week 15 mean: {hist_w15:.1f}%\n"
            f"  Week 40 mean: {hist_w40:.1f}%"
        )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )

    plt.tight_layout()
    out = paper_output_path / output_name
    fig.savefig(out, format="pdf", bbox_inches="tight")
    logger.info(f"Saved reservoir trajectory to {out}")
    return fig

# Default script config
SCENARIOS = {
    "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF": "B",
}
MEAN_ONLY_SCENARIOS = {
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF": "B+",
}
OUTPUT_NAME = "reservoir_trajectory_baseline_comparison.pdf"

fig = plot_reservoir_trajectory(
    scenarios_config=SCENARIOS,
    mean_only_scenarios=MEAN_ONLY_SCENARIOS,
    output_name=OUTPUT_NAME,
)


SCENARIOS = {
    "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA",
    # "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-BA",
    "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA",
}
MEAN_ONLY_SCENARIOS = {
    # "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF": "B",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA+",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA+",
}
OUTPUT_NAME = "reservoir_trajectory_n_ow_comparison.pdf"
fig = plot_reservoir_trajectory(
    scenarios_config=SCENARIOS,
    mean_only_scenarios=MEAN_ONLY_SCENARIOS,
    output_name=OUTPUT_NAME,
    percentile_range = (5.0, 95.0),
    include_historical=True
)
