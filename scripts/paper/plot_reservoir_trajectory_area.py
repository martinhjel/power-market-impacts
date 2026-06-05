"""
Plot historical reservoir trajectory for configurable scenarios by Norwegian area.
Supports per-scenario mean-only mode (no percentile fill) and optional historical overlay.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        logger.info("Loaded historical reservoir data from NVE (NO aggregate)")
        return {"mean": hist_mean, "p10": hist_p10, "p90": hist_p90}
    except Exception as e:
        logger.warning(f"Failed to load historical reservoir data: {e}")
        return None


def plot_reservoir_trajectory_area(
    scenarios_config: dict[str, str],
    mean_only_scenarios: list[str] | set[str] | tuple[str, ...] | dict[str, str] | None = None,
    output_name: str = "reservoir_trajectory_area.pdf",
    percentile_range: tuple[float, float] = (10.0, 90.0),
    include_historical: bool = True,
    areas: list[str] | tuple[str, ...] | None = None,
) -> plt.Figure:
    """
    Plot reservoir trajectory by area for selected scenarios.

    Args:
        scenarios_config: Mapping {scenario_folder_name: plot_label}.
        mean_only_scenarios:
            - list/set/tuple of selectors (folder name or plot label), or
            - dict {scenario_folder_name: plot_label} for additional mean-only scenarios.
        output_name: Output PDF filename in paper output folder.
        percentile_range: Lower/upper percentile for scenario range fill (e.g. (10, 90)).
        include_historical: Whether to load and plot historical NO trajectory/range.
        areas: Areas to plot. Defaults to NO1..NO5.

    Returns:
        Matplotlib figure handle.
    """
    p_low, p_high = percentile_range
    if not (0 <= p_low < p_high <= 100):
        raise ValueError(f"Invalid percentile_range={percentile_range}. Expected 0 <= low < high <= 100.")

    selected_areas = list(areas) if areas is not None else NO_BUSBARS
    if not selected_areas:
        raise ValueError("No areas selected")

    base_path = _resolve_base_path()
    output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
    paper_output_path = output_path / "paper"
    paper_output_path.mkdir(parents=True, exist_ok=True)

    scenarios_to_plot = dict(scenarios_config)
    selectors: set[str] = set()

    if isinstance(mean_only_scenarios, dict):
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
    logger.info(f"Selected areas: {selected_areas}")
    if selectors:
        logger.info(f"Mean-only selectors: {sorted(selectors)}")

    scenario_data: dict[str, dict[str, dict[str, pd.Series]]] = {}
    scenario_meta: dict[str, dict[str, object]] = {}
    n_steps = None

    for scenario_name, label in scenarios_to_plot.items():
        if scenario_name not in scenarios:
            logger.warning(f"Scenario {scenario_name} not found")
            continue

        scenario = scenarios[scenario_name]
        logger.info(f"Processing scenario: {label}")
        area_data: dict[str, dict[str, pd.Series]] = {}

        for area in selected_areas:
            total_reservoir = None
            max_volume = 0.0
            try:
                val = scenario.get_reservoir_for_busbar(area)
                total_reservoir = val
                max_volume += val.max().max()
            except Exception as e:
                logger.warning(f"  Failed to get reservoir for {area}: {e}")
                continue

            if total_reservoir is None or max_volume == 0:
                logger.warning(f"  No reservoir data collected for {area} in {label}")
                continue

            total_pct = (total_reservoir / max_volume) * 100
            area_data[area] = {
                "mean": total_pct.mean(axis=1),
                "p_low": total_pct.quantile(p_low / 100.0, axis=1),
                "p_high": total_pct.quantile(p_high / 100.0, axis=1),
            }
            if n_steps is None:
                n_steps = len(area_data[area]["mean"])

        if area_data:
            scenario_data[label] = area_data
            scenario_meta[label] = {
                "scenario_name": scenario_name,
                "mean_only": (label in selectors) or (scenario_name in selectors),
            }

    if not scenario_data:
        raise RuntimeError("No scenario data collected")
    if n_steps is None:
        raise RuntimeError("No timestep data collected")

    time_steps = np.arange(n_steps)
    week_index = (time_steps % 52) + 1
    historical_data = _load_historical_data(base_path, n_steps) if include_historical else None

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

    n_areas = len(selected_areas)
    ncols = min(3, n_areas)
    nrows = int(np.ceil(n_areas / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, area in enumerate(selected_areas):
        ax = axes_flat[i]

        if historical_data:
            ax.fill_between(
                time_steps,
                historical_data["p10"],
                historical_data["p90"],
                alpha=0.15,
                color="gray",
                label="Historical NO p10-p90 range",
            )
            ax.plot(
                time_steps,
                historical_data["mean"],
                color="black",
                linewidth=1.8,
                label="Historical NO Mean",
                linestyle="-",
                alpha=0.7,
            )

        has_area_data = False
        stats_lines = []
        for label, data_by_area in scenario_data.items():
            if area not in data_by_area:
                continue
            has_area_data = True
            data = data_by_area[area]

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
                linewidth=2.3,
                label=f"{label} Mean",
                linestyle=linestyle,
            )

            mean_values = np.asarray(data["mean"], dtype=float)
            w15 = float(np.nanmean(mean_values[week_index == 15])) if np.any(week_index == 15) else float("nan")
            w40 = float(np.nanmean(mean_values[week_index == 40])) if np.any(week_index == 40) else float("nan")
            stats_lines.append(
                f"{label}: Mean {mean_values.mean():.1f}% | W15 {w15:.1f}% | W40 {w40:.1f}%"
            )

        ax.set_title(area, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Step (weeks)", fontsize=10)
        ax.set_ylabel("Reservoir Filling (%)", fontsize=10)
        ax.set_ylim(bottom=0, top=100)
        ax.grid(True, alpha=0.3)

        if has_area_data:
            ax.legend(loc="best", framealpha=0.95, fontsize=8, ncol=1)
            if stats_lines:
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(stats_lines),
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=7.5,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )

    for j in range(n_areas, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Reservoir Trajectory by Norwegian Area", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = paper_output_path / output_name
    fig.savefig(out, format="pdf", bbox_inches="tight")
    logger.info(f"Saved reservoir trajectory by area to {out}")
    return fig


# Default script config
SCENARIOS = {
    "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF": "B",
    "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF": "B+",
}
MEAN_ONLY_SCENARIOS = {}
AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
OUTPUT_NAME = "reservoir_trajectory_area_baseline_comparison.pdf"

fig = plot_reservoir_trajectory_area(
    scenarios_config=SCENARIOS,
    mean_only_scenarios=MEAN_ONLY_SCENARIOS,
    output_name=OUTPUT_NAME,
    percentile_range=(10.0, 90.0),
    include_historical=True,
    areas=AREAS,
)
