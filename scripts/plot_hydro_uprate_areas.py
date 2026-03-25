"""
Plot hydropower and price duration curves for multiple scenario comparisons in NO1-NO5.
Includes the requested N-BA vs N-BA+ comparison in addition to other outputs.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import ScenarioStyler, load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
SHORT_TO_SCENARIO = {
    "N-BA": "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "N-BA+": "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "N-LLPS": "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "N-LLPS+": "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OW-BA": "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "OW-BA+": "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "OW-LLPS": "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "OW-LLPS+": "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "OWN-BA+": "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OWN-BA": "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
}

def _resolve_base_path() -> Path:
    p = Path.cwd()
    if p.name == "paper":
        return p.parent.parent
    if p.name == "scripts":
        return p.parent
    return p


base_path = _resolve_base_path()
paper_output_path = base_path / OUTPUT_DIR / MODEL_FOLDER / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)

area_styler = ScenarioStyler()


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


SHOW_IN_NOTEBOOK = _running_in_notebook()


def _sorted_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values)[::-1]
    return np.linspace(0, 1, len(sorted_values)), sorted_values


def _plot_panel(
    ax: plt.Axes,
    scenarios: dict,
    scenario_labels: dict[str, str],
    scenario_linestyles: dict[str, str],
    value_getter,
    y_label: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    for scenario_name, scenario in scenarios.items():
        short_name = scenario_labels.get(scenario_name, scenario_name)
        linestyle = scenario_linestyles.get(short_name, "solid")
        linewidth = 2.6 if short_name.endswith("+") else 2.0

        for area in AREAS:
            try:
                series = value_getter(scenario, area)
                x, y = _sorted_curve(series.values.flatten())
                style = area_styler.mpl_style(area)
                ax.plot(
                    x,
                    y,
                    label=f"{area} ({short_name})",
                    color=style.color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=0.9,
                )
            except Exception as e:
                logger.warning(f"Failed plotting {area} for {short_name}: {e}")

    ax.set_xlabel("Fraction of hours", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")


def _spearman_from_series(y1: np.ndarray, y2: np.ndarray) -> float:
    if len(y1) == 0 or len(y2) == 0:
        return float("nan")
    n = min(len(y1), len(y2))
    if n == 0:
        return float("nan")
    return float(pd.Series(y1[:n]).corr(pd.Series(y2[:n]), method="spearman"))


def _scenario_pairs_for_spearman_table() -> list[tuple[str, str]]:
    # Using the exact comparisons requested by user.
    # "N-BA-OW-BA+" is interpreted as "N-BA vs OW-BA+".
    return [
        ("N-BA", "N-BA+"),
        ("N-LLPS", "N-BA+"),
        ("N-BA", "OW-BA+"),
        ("OW-BA", "OW-BA+"),
        ("OW-LLPS", "OW-BA+"),
    ]


def _pair_header(left: str, right: str) -> str:
    return f"{right} - {left}"


def _scenario_pairs_for_p99_table() -> list[tuple[str, str]]:
    # Requested P99 table columns:
    # N-BA+ - N-BA
    # N-LLPS+ - N-LLPS
    # OW-BA+ - N-BA
    # OWN-BA+ - N-BA
    # OW-BA+ - OW-BA
    # OW-LLPS+ - OW-LLPS
    return [
        ("N-BA", "N-BA+"),
        ("N-LLPS", "N-BA+"),
        ("OW-BA", "OW-BA+"),
        ("N-BA", "OW-BA"),
        ("N-BA", "OW-BA+"),
        ("N-BA+", "OW-BA+"),
    ]


def _compute_pairwise_spearman_rows() -> list[dict]:
    rows = []
    pairs = _scenario_pairs_for_spearman_table()
    needed_short = sorted({s for p in pairs for s in p})
    scenario_paths = {SHORT_TO_SCENARIO[s]: base_path / f"ltm_output/{MODEL_FOLDER}/{SHORT_TO_SCENARIO[s]}" for s in needed_short}
    scenarios = load_scenarios(scenario_paths)

    short_to_loaded = {short: scenarios.get(long_name) for short, long_name in SHORT_TO_SCENARIO.items() if long_name in scenarios}

    for left, right in pairs:
        s_left = short_to_loaded.get(left)
        s_right = short_to_loaded.get(right)
        for area in AREAS:
            hydro_rho = float("nan")
            price_rho = float("nan")
            if s_left is not None and s_right is not None:
                try:
                    hydro_left = s_left.get_hydro_production_for_busbar(area).values.flatten()
                    hydro_right = s_right.get_hydro_production_for_busbar(area).values.flatten()
                    hydro_rho = _spearman_from_series(hydro_left, hydro_right)
                except Exception:
                    pass
                try:
                    price_left = s_left.get_prices_for_busbar(area).values.flatten()
                    price_right = s_right.get_prices_for_busbar(area).values.flatten()
                    price_rho = _spearman_from_series(price_left, price_right)
                except Exception:
                    pass
            rows.append(
                {
                    "pair": _pair_header(left, right),
                    "area": area,
                    "hydro_spearman": hydro_rho,
                }
            )
    return rows


def _write_spearman_latex_table(rows: list[dict], output_file: Path) -> None:
    pairs = [_pair_header(left, right) for left, right in _scenario_pairs_for_spearman_table()]
    hydro_by_area_pair: dict[str, dict[str, float]] = {area: {} for area in AREAS}
    for row in rows:
        hydro_by_area_pair[row["area"]][row["pair"]] = row["hydro_spearman"]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Hydropower Spearman rank correlation by area for selected scenario comparisons}",
        "\\label{tab:spearman_area_pairs}",
        "\\begin{tabular}{l" + "c" * len(pairs) + "}",
        "\\hline",
        "Area & " + " & ".join(pairs) + " \\\\",
        "\\hline",
    ]
    for area in AREAS:
        vals = []
        for pair in pairs:
            v = hydro_by_area_pair.get(area, {}).get(pair, float("nan"))
            vals.append(f"{v:.4f}" if not np.isnan(v) else "n/a")
        lines.append(f"{area} & " + " & ".join(vals) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved Spearman LaTeX table to {output_file}")


def _compute_hydro_percentile_diff_rows_for_pairs(
    pairs: list[tuple[str, str]],
    percentile: float = 99.0,
) -> list[dict]:
    """
    Compute hydro duration percentile-level difference per area for requested pairs.
    Difference convention: Pxx(right scenario) - Pxx(left scenario), in MW.
    """
    if percentile < 0 or percentile > 100:
        raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

    rows = []
    needed_short = sorted({s for p in pairs for s in p})
    scenario_paths = {SHORT_TO_SCENARIO[s]: base_path / f"ltm_output/{MODEL_FOLDER}/{SHORT_TO_SCENARIO[s]}" for s in needed_short}
    scenarios = load_scenarios(scenario_paths)
    short_to_loaded = {short: scenarios.get(long_name) for short, long_name in SHORT_TO_SCENARIO.items() if long_name in scenarios}

    for left, right in pairs:
        s_left = short_to_loaded.get(left)
        s_right = short_to_loaded.get(right)
        for area in AREAS:
            delta_pxx = float("nan")
            if s_left is not None and s_right is not None:
                try:
                    hydro_left = s_left.get_hydro_production_for_busbar(area).values.flatten()
                    hydro_right = s_right.get_hydro_production_for_busbar(area).values.flatten()
                    p_left = float(np.nanpercentile(hydro_left, percentile))
                    p_right = float(np.nanpercentile(hydro_right, percentile))
                    delta_pxx = p_right - p_left
                except Exception:
                    pass
            rows.append({"pair": _pair_header(left, right), "area": area, "delta_percentile_mw": delta_pxx})
    return rows


def _write_percentile_diff_latex_table(
    rows: list[dict],
    output_file: Path,
    pairs: list[tuple[str, str]],
    percentile: float,
) -> None:
    pairs_headers = [_pair_header(left, right) for left, right in pairs]
    by_area_pair: dict[str, dict[str, float]] = {area: {} for area in AREAS}
    for row in rows:
        by_area_pair[row["area"]][row["pair"]] = row["delta_percentile_mw"]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Hydropower percentile difference by area (P{percentile:.0f} right minus left, MW)}}",
        "\\label{tab:hydro_p99_diff_area_pairs}",
        "\\begin{tabular}{l" + "c" * len(pairs_headers) + "}",
        "\\hline",
        "Area & " + " & ".join(pairs_headers) + " \\\\",
        "\\hline",
    ]
    for area in AREAS:
        vals = []
        for pair in pairs_headers:
            v = by_area_pair.get(area, {}).get(pair, float("nan"))
            vals.append(f"{v:+.1f}" if not np.isnan(v) else "n/a")
        lines.append(f"{area} & " + " & ".join(vals) + " \\\\")
    sum_vals = []
    for pair in pairs_headers:
        col_vals = [by_area_pair.get(area, {}).get(pair, float("nan")) for area in AREAS]
        valid = [v for v in col_vals if not np.isnan(v)]
        if valid:
            sum_vals.append(f"{sum(valid):+.1f}")
        else:
            sum_vals.append("n/a")
    lines.append("\\hline")
    lines.append("Sum & " + " & ".join(sum_vals) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved hydro P99-difference LaTeX table to {output_file}")


def _add_spearman_box(
    ax: plt.Axes,
    scenarios: dict,
    scenario_names: list[str],
    scenario_labels: dict[str, str],
    value_getter,
) -> None:
    # Spearman box is defined for pairwise comparison bundles.
    if len(scenario_names) != 2:
        return
    if scenario_names[0] not in scenarios or scenario_names[1] not in scenarios:
        return

    scenario_a = scenarios[scenario_names[0]]
    scenario_b = scenarios[scenario_names[1]]
    label_a = scenario_labels.get(scenario_names[0], scenario_names[0])
    label_b = scenario_labels.get(scenario_names[1], scenario_names[1])

    lines = [f"Spearman hourly ({label_a} vs {label_b})"]
    for area in AREAS:
        try:
            y1 = value_getter(scenario_a, area).values.flatten()
            y2 = value_getter(scenario_b, area).values.flatten()
            rho = _spearman_from_series(y1, y2)
            lines.append(f"{area}: {rho:.3f}")
        except Exception:
            lines.append(f"{area}: n/a")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.85),
    )


def _save_comparison_bundle(
    *,
    scenario_names: list[str],
    scenario_labels: dict[str, str],
    scenario_linestyles: dict[str, str],
    title: str,
    file_stem: str,
    price_ylim: tuple[float, float] = (0, 200),
) -> None:
    scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in scenario_names}
    scenarios = load_scenarios(scenario_paths)

    if not scenarios:
        logger.warning(f"No scenarios loaded for bundle: {file_stem}")
        return

    output_combined = paper_output_path / f"hydro_price_duration_{file_stem}.pdf"
    output_latex = paper_output_path / f"mean_prices_table_{file_stem}.tex"

    # Combined figure
    fig, (ax_hydro, ax_price) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    _plot_panel(
        ax_hydro,
        scenarios,
        scenario_labels,
        scenario_linestyles,
        value_getter=lambda s, a: s.get_hydro_production_for_busbar(a),
        y_label="Hydropower Generation (MW)",
        title="Hydropower Duration",
    )
    _plot_panel(
        ax_price,
        scenarios,
        scenario_labels,
        scenario_linestyles,
        value_getter=lambda s, a: s.get_prices_for_busbar(a),
        y_label="Price (EUR/MWh)",
        title="Price Duration",
        ylim=price_ylim,
    )
    plt.tight_layout()
    fig.savefig(output_combined, format="pdf", bbox_inches="tight")
    if SHOW_IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

    # Mean price LaTeX table
    labels_in_order = [scenario_labels[s] for s in scenario_names if s in scenario_labels]
    price_data: dict[str, dict[str, float]] = {}
    for s_name, scenario in scenarios.items():
        label = scenario_labels.get(s_name, s_name)
        for area in AREAS:
            try:
                mean_price = float(scenario.get_prices_for_busbar(area).values.mean())
                price_data.setdefault(area, {})[label] = mean_price
            except Exception as e:
                logger.warning(f"Failed mean price for {area} in {label}: {e}")

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Mean Power Prices by Area ({title})}}",
        f"\\label{{tab:mean_prices_{file_stem}}}",
        "\\begin{tabular}{l" + "c" * len(labels_in_order) + "}",
        "\\hline",
        "Area & " + " & ".join(f"{lbl} (EUR/MWh)" for lbl in labels_in_order) + " \\\\",
        "\\hline",
    ]
    for area in AREAS:
        vals = [price_data.get(area, {}).get(lbl, float("nan")) for lbl in labels_in_order]
        latex_lines.append(area + " & " + " & ".join(f"{v:.2f}" for v in vals) + " \\\\")
    latex_lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    with open(output_latex, "w") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"Saved bundle: {file_stem}")
    logger.info(f"  {output_combined}")
    logger.info(f"  {output_latex}")


# 1) Baseline vs Baseline Uprate (original-type comparison)
_save_comparison_bundle(
    scenario_names=[
        "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
        "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF",
    ],
    scenario_labels={
        "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF": "B",
        "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF": "B+",
    },
    scenario_linestyles={"B": "dashed", "B+": "solid"},
    title="Hydropower and Price Duration Curves - Baseline vs Baseline Uprate (NO1-NO5)",
    file_stem="NO_areas_baseline_vs_uprate",
    price_ylim=(0, 130),
)

# 2) N-BA+ vs OW-BA+ (existing additional comparison)
_save_comparison_bundle(
    scenario_names=[
        "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    ],
    scenario_labels={
        "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA+",
        "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA+",
    },
    scenario_linestyles={"N-BA+": "solid", "OW-BA+": "dotted"},
    title="Hydropower and Price Duration Curves - N-BA+ vs OW-BA+ (NO1-NO5)",
    file_stem="NO_areas_N_OW_uprate",
    price_ylim=(0, 120),
)

# 3) Requested additional comparison: N-BA vs N-BA+
_save_comparison_bundle(
    scenario_names=[
        "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    ],
    scenario_labels={
        "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA",
        "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA+",
    },
    scenario_linestyles={"N-BA": "dashed", "N-BA+": "solid"},
    title="Hydropower and Price Duration Curves - N-BA vs N-BA+ (NO1-NO5)",
    file_stem="NO_areas_N_BA_vs_N_BA_plus",
    price_ylim=(0, 200),
)

# 4) N-BA vs OW-BA+ (existing additional comparison)
_save_comparison_bundle(
    scenario_names=[
        "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    ],
    scenario_labels={
        "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA",
        "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA+",
    },
    scenario_linestyles={"N-BA": "solid", "OW-BA+": "dotted"},
    title="Hydropower and Price Duration Curves - N-BA vs OW-BA+ (NO1-NO5)",
    file_stem="NO_areas_N_BA_OW_BA_plus",
    price_ylim=(0, 120),
)

# 5) N-LLPS vs OW-BA+ (existing additional comparison)
_save_comparison_bundle(
    scenario_names=[
        "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    ],
    scenario_labels={
        "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-LLPS",
        "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA+",
    },
    scenario_linestyles={"N-LLPS": "solid", "OW-BA+": "dotted"},
    title="Hydropower and Price Duration Curves - N-LLPS vs OW-BA+ (NO1-NO5)",
    file_stem="NO_areas_N_LLPS_OW_BA_plus",
    price_ylim=(0, 200),
)

# 6) Requested additional comparison: N-LLPS vs N-BA+
_save_comparison_bundle(
    scenario_names=[
        "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    ],
    scenario_labels={
        "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-LLPS",
        "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA+",
    },
    scenario_linestyles={"N-LLPS": "dashed", "N-BA+": "solid"},
    title="Hydropower and Price Duration Curves - N-LLPS vs N-BA+ (NO1-NO5)",
    file_stem="NO_areas_N_LLPS_vs_N_BA_plus",
    price_ylim=(0, 200),
)

# 7) N-BA vs OW-BA (existing additional comparison)
_save_comparison_bundle(
    scenario_names=[
        "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
        "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    ],
    scenario_labels={
        "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA",
        "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA",
    },
    scenario_linestyles={"N-BA": "solid", "OW-BA": "dotted"},
    title="Hydropower and Price Duration Curves - N-BA vs OW-BA (NO1-NO5)",
    file_stem="NO_areas_N_BA_OW_BA",
    price_ylim=(0, 120),
)

# Pairwise Spearman LaTeX table for requested comparisons
spearman_rows = _compute_pairwise_spearman_rows()
_write_spearman_latex_table(
    spearman_rows, paper_output_path / "spearman_correlations_selected_pairs.tex"
)

base_p99_pairs = _scenario_pairs_for_spearman_table()
base_percentile_rows = _compute_hydro_percentile_diff_rows_for_pairs(base_p99_pairs, percentile=99.0)
_write_percentile_diff_latex_table(
    base_percentile_rows,
    paper_output_path / "hydro_p99_diff_selected_pairs.tex",
    base_p99_pairs,
    percentile=99.0,
)

percentile = 99
requested_p99_pairs = _scenario_pairs_for_p99_table()
requested_percentile_rows = _compute_hydro_percentile_diff_rows_for_pairs(requested_p99_pairs, percentile=percentile)
_write_percentile_diff_latex_table(
    requested_percentile_rows,
    paper_output_path / f"hydro_p{percentile}_diff_selected_pairs_requested.tex",
    requested_p99_pairs,
    percentile=float(percentile),
)

# Backward-compatible alias variables kept for readability where this file
# has historically referred to "p99" outputs.
base_p99_rows = base_percentile_rows
requested_p99_rows = requested_percentile_rows
_ = base_p99_rows, requested_p99_rows
