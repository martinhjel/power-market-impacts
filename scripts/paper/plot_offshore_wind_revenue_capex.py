"""
Plot offshore wind net revenue and inferred CAPEX from calculated revenue results.

Reads:
- visualizations/<MODEL_FOLDER>/paper/nuclear_offshore_revenue.csv

Writes:
- visualizations/<MODEL_FOLDER>/paper/offshore_wind_revenue_capex.pdf
- visualizations/<MODEL_FOLDER>/paper/offshore_wind_revenue_capex.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import logger

MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
SCENARIO_MAP = {
    "N-LLPS+": "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-LLPS+": "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-LLPS+": "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "N-BA+": "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-BA+": "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-BA+": "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "N-LLPS": "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-LLPS": "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-LLPS": "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "N-BA": "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-BA": "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-BA": "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "SMR300-BA": "SMR300BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR300-LLPS": "SMR300LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF",
    "SMR600-BA": "SMR600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR600-LLPS": "SMR600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_600NO1-600NO2-600NO3-600NO4-600NO5NUKE_NoneOFF",
    "SMR900-BA": "SMR900BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR900-LLPS": "SMR900LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_900NO1-900NO2-900NO3-900NO4-900NO5NUKE_NoneOFF",
    "SMR1200-BA": "SMR1200BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1200-LLPS": "SMR1200LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1200NO1-1200NO2-1200NO3-1200NO4-1200NO5NUKE_NoneOFF",
    "SMR1600-BA": "SMR1600BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    "SMR1600-LLPS": "SMR1600LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_1600NO1-1600NO2-1600NO3-1600NO4-1600NO5NUKE_NoneOFF",
    "LMR2000-BA": "LMR2000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR2000-LLPS": "LMR2000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_2000NO1-2000NO2NUKE_NoneOFF",
    "LMR3000-BA": "LMR3000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR3000-LLPS": "LMR3000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_3000NO1-3000NO2NUKE_NoneOFF",
    "LMR4000-BA": "LMR4000BA_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
    "LMR4000-LLPS": "LMR4000LLPS_30TWh_FalseHYD_FalseFF_LLPSLOAD_30.00TWH_4000NO1-4000NO2NUKE_NoneOFF",
}
SCENARIO_ORDER = list(SCENARIO_MAP.keys())


def _weighted_average(series: pd.Series, weights: pd.Series) -> float:
    positive_weights = weights.clip(lower=0.0)
    if positive_weights.sum() <= 0:
        return float(series.mean()) if len(series) > 0 else 0.0
    return float((series * positive_weights).sum() / positive_weights.sum())


def _normalize_scenario(value: str) -> str:
    long_to_short = {v: k for k, v in SCENARIO_MAP.items()}
    return long_to_short.get(value, value)



base_path = Path.cwd()
if base_path.name == "paper":
    base_path = base_path.parent.parent
elif base_path.name == "scripts":
    base_path = base_path.parent

output_path = base_path / OUTPUT_DIR / MODEL_FOLDER / "paper"
csv_path = output_path / "nuclear_offshore_revenue.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Missing input file: {csv_path}")

df = pd.read_csv(csv_path)
df["scenario"] = df["scenario"].map(_normalize_scenario)
wind_df = df[df["technology"] == "Offshore Wind"].copy()
if wind_df.empty:
    raise ValueError("No offshore wind rows found in nuclear_offshore_revenue.csv")

scenario_order = [s for s in SCENARIO_ORDER if s in set(wind_df["scenario"])]
agg_rows = []
for scenario in scenario_order:
    group = wind_df[wind_df["scenario"] == scenario]
    agg_rows.append(
        {
            "scenario": scenario,
            "net_revenue_meur": float(group["net_revenue_meur"].sum()),
            "breakeven_capex_eur_per_kw": _weighted_average(
                group["breakeven_capex_eur_per_kw"], group["generation_gwh"]
            ),
        }
    )

agg_df = pd.DataFrame(agg_rows)
if agg_df.empty:
    raise ValueError("No offshore wind aggregates could be computed from input CSV")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(agg_df["scenario"], agg_df["net_revenue_meur"], color="#1f77b4", alpha=0.85)
axes[0].set_title("Offshore Wind Net Revenue")
axes[0].set_ylabel("M EUR/year")
axes[0].set_xlabel("Scenario")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(agg_df["scenario"], agg_df["breakeven_capex_eur_per_kw"], color="#ff7f0e", alpha=0.85)
axes[1].set_title("Offshore Wind Inferred CAPEX (FID)")
axes[1].set_ylabel("EUR/kW")
axes[1].set_xlabel("Scenario")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", alpha=0.3)

fig.tight_layout()
output_pdf = output_path / "offshore_wind_revenue_capex.pdf"
output_png = output_path / "offshore_wind_revenue_capex.png"
fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
fig.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close(fig)

logger.info(f"Saved offshore wind revenue/CAPEX figure to {output_pdf}")
logger.info(f"Saved offshore wind revenue/CAPEX figure to {output_png}")

