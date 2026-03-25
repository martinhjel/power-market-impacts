"""
Plot evolution of mean power prices in Norwegian areas for BA scenarios.

X-axis follows the nuclear-capacity axis used in visualize_smr_lmr_surplus.py:
total installed nuclear capacity (MW) for baseline, SMR, and LMR BA cases.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import logger

MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
AREAS = ["NO2", "NO1", "NO5", "NO3", "NO4"]
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

SCENARIOS = {
    "BASELINE": "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF",
    "SMR300_BA": "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR600_BA": "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR900_BA": "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR1200_BA": "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1600_BA": "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    "LMR2000_BA": "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR3000_BA": "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR4000_BA": "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
}


def _scenario_capacity_mw(short_name: str) -> int:
    if short_name == "BASELINE":
        return 0
    if short_name.startswith("SMR"):
        return 5 * int(short_name.split("_")[0][3:])
    if short_name.startswith("LMR"):
        return 2 * int(short_name.split("_")[0][3:])
    return 0


def _load_process_df(scenario_path: Path) -> pd.DataFrame | None:
    process_file = scenario_path / "results" / "processed" / "market_dispatch.parquet"
    if not process_file.exists():
        logger.warning(f"Missing process file: {process_file}")
        return None
    return pd.read_parquet(process_file)


def _load_export_df(scenario_path: Path) -> pd.DataFrame | None:
    export_file = scenario_path / "results" / "processed" / "market_dispatch.pkl"
    if not export_file.exists():
        logger.warning(f"Missing export file: {export_file}")
        return None
    return pd.read_pickle(export_file)


def _marker_for_scenario(short_name: str) -> str:
    if short_name.startswith("SMR"):
        return "*"
    if short_name.startswith("LMR"):
        return "s"
    return "o"


def _scenario_family(short_name: str) -> str:
    if short_name.startswith("SMR"):
        return "SMR"
    if short_name.startswith("LMR"):
        return "LMR"
    return "B30"


base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER / "paper"
output_path.mkdir(parents=True, exist_ok=True)

rows: list[dict[str, float | int | str]] = []
vwap_rows: list[dict[str, float | int | str]] = []
for short_name, scenario_name in SCENARIOS.items():
    scenario_path = base_path / "ltm_output" / MODEL_FOLDER / scenario_name
    if not scenario_path.exists():
        logger.warning(f"Scenario path does not exist: {scenario_path}")
        continue

    process_df = _load_process_df(scenario_path)
    if process_df is None:
        continue
    export_df = _load_export_df(scenario_path)

    for area in AREAS:
        if area not in process_df.index.get_level_values("area"):
            continue
        if export_df is None or area not in export_df.index.get_level_values("area"):
            continue
        area_joined = export_df.xs(area, level="area")[["load"]].join(
            process_df.xs(area, level="area")[["market_price"]], how="inner"
        )
        area_load = area_joined["load"].sum()
        if area_load <= 0:
            continue
        rows.append(
            {
                "scenario": short_name,
                "capacity_mw": _scenario_capacity_mw(short_name),
                "area": area,
                "mean_price": (area_joined["load"] * area_joined["market_price"]).sum() / area_load,
            }
        )

    if export_df is not None:
        weighted_price_sum = 0.0
        load_sum = 0.0
        for area in NO_AREAS:
            if area not in process_df.index.get_level_values("area"):
                continue
            if area not in export_df.index.get_level_values("area"):
                continue
            df_area = export_df.xs(area, level="area")[["load"]].join(
                process_df.xs(area, level="area")[["market_price"]], how="inner"
            )
            weighted_price_sum += (df_area["load"] * df_area["market_price"]).sum()
            load_sum += df_area["load"].sum()
        if load_sum > 0:
            vwap_rows.append(
                {
                    "scenario": short_name,
                    "capacity_mw": _scenario_capacity_mw(short_name),
                    "family": _scenario_family(short_name),
                    "vwap_price": weighted_price_sum / load_sum,
                }
            )

if not rows:
    raise ValueError("No BA scenario market price data found.")

df = pd.DataFrame(rows)
df_vwap = pd.DataFrame(vwap_rows)
scenario_order = [
    "BASELINE",
    "SMR300_BA",
    "SMR600_BA",
    "LMR2000_BA",
    "SMR900_BA",
    "SMR1200_BA",
    "LMR3000_BA",
    "SMR1600_BA",
    "LMR4000_BA",
]
scenario_order = [s for s in scenario_order if s in df["scenario"].unique()]

scenario_x: dict[str, float] = {s: float(_scenario_capacity_mw(s)) for s in scenario_order}

fig, ax = plt.subplots(figsize=(12, 4))
area_offsets = np.linspace(-35.0, 35.0, len(AREAS))
area_colors = {area: plt.cm.tab10(i) for i, area in enumerate(AREAS)}

for i, area in enumerate(AREAS):
    area_df = df[df["area"] == area].set_index("scenario")
    area_color = area_colors[area]

    smr_labels = [s for s in scenario_order if s in area_df.index and s.startswith("SMR")]
    lmr_labels = [s for s in scenario_order if s in area_df.index and s.startswith("LMR")]
    b_labels = [s for s in scenario_order if s in area_df.index and s == "BASELINE"]

    if smr_labels:
        smr_x = [scenario_x[s] + area_offsets[i] for s in smr_labels]
        smr_y = [area_df.loc[s, "mean_price"] for s in smr_labels]
        ax.plot(smr_x, smr_y, color=area_color, linestyle="-", linewidth=1.8, alpha=0.95)
        ax.scatter(smr_x, smr_y, color=area_color, marker="*", s=120, zorder=3)

    if lmr_labels:
        lmr_x = [scenario_x[s] + area_offsets[i] for s in lmr_labels]
        lmr_y = [area_df.loc[s, "mean_price"] for s in lmr_labels]
        ax.plot(lmr_x, lmr_y, color=area_color, linestyle="--", linewidth=1.8, alpha=0.95)
        ax.scatter(lmr_x, lmr_y, color=area_color, marker="s", s=70, zorder=3)

    if b_labels:
        x0 = scenario_x["BASELINE"] + area_offsets[i]
        y0 = area_df.loc["BASELINE", "mean_price"]
        ax.plot([x0 - 25, x0 + 25], [y0, y0], color=area_color, linestyle=":", linewidth=1.8, alpha=0.95)
        ax.scatter([x0], [y0], color=area_color, marker="o", s=55, zorder=3)

if not df_vwap.empty:
    smr_df = df_vwap[df_vwap["family"] == "SMR"].sort_values("capacity_mw")
    lmr_df = df_vwap[df_vwap["family"] == "LMR"].sort_values("capacity_mw")

    if not smr_df.empty:
        ax.plot(
            [scenario_x[s] for s in smr_df["scenario"]],
            smr_df["vwap_price"],
            color="black",
            linewidth=2.6,
            linestyle="-",
            marker="*",
            markersize=11,
            zorder=4,
        )
    if not lmr_df.empty:
        ax.plot(
            [scenario_x[s] for s in lmr_df["scenario"]],
            lmr_df["vwap_price"],
            color="dimgray",
            linewidth=2.6,
            linestyle="--",
            marker="s",
            markersize=7.5,
            zorder=4,
        )

ax.set_xlabel("Total Installed Nuclear Capacity (MW)")
ax.set_ylabel("Mean Power Price (EUR/MWh)")
ax.set_title("Mean Power Price by Area vs Nuclear Capacity (BA Scenarios)")
ax.grid(True, alpha=0.3)
unique_caps = sorted({_scenario_capacity_mw(s) for s in scenario_order})
ax.set_xticks(unique_caps)
ax.set_xticklabels([str(cap) for cap in unique_caps])

area_legend = [
    Line2D([0], [0], color=area_colors[area], linestyle="-", linewidth=2, label=area) for area in AREAS
]
first_legend = ax.legend(
    handles=area_legend,
    title="Area",
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    frameon=True,
)
ax.add_artist(first_legend)
type_legend = [
    Line2D([0], [0], color="black", linestyle="-", marker="*", markersize=9, label="SMR"),
    Line2D([0], [0], color="black", linestyle="--", marker="s", markersize=7, label="LMR"),
    Line2D([0], [0], color="black", linestyle=":", marker="o", markersize=6, label="B30"),
]
second_legend = ax.legend(handles=type_legend, title="Case Type", loc="upper right")
ax.add_artist(second_legend)
if not df_vwap.empty:
    vwap_legend = [
        Line2D([0], [0], color="black", linewidth=2.6, linestyle="-", marker="*", markersize=10, label="VWAP SMR"),
        Line2D([0], [0], color="dimgray", linewidth=2.6, linestyle="--", marker="s", markersize=7, label="VWAP LMR"),
    ]
    third_legend = ax.legend(handles=vwap_legend, title="All Areas VWAP", loc="lower right")
    ax.add_artist(third_legend)

plt.tight_layout()
output_png = output_path / "mean_power_prices_ba_capacity_line.png"
output_pdf = output_path / "mean_power_prices_ba_capacity_line.pdf"
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.savefig(output_pdf, bbox_inches="tight")
logger.info(f"Saved figure: {output_png}")
logger.info(f"Saved figure: {output_pdf}")
