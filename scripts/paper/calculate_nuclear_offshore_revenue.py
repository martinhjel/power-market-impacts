"""
Calculate revenue for nuclear and offshore wind resources in BA_N, BA_OW, and BA_OWN scenarios.

This script:
1. Loads scenario results for N (Nuclear), OW (Offshore Wind), and OWN (Offshore Wind + Nuclear) cases
2. Extracts nuclear generation using market steps (similar to viz_nuclear_factors.py)
3. Extracts offshore wind generation from wind objects in areas SNII, UN, and VVD
4. Calculates revenue = generation * price for each technology
5. Computes value factors and capacity factors
6. Computes break-even CAPEX at construction start (FID)
7. Outputs results to CSV and creates visualizations
"""

import sys
from pathlib import Path

import logging
from typing import Tuple

import pandas as pd

# Add workspace root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import logger
from scripts.paper.processed_dispatch import load_processed_dispatch_data

# Set logger level to DEBUG to see debug messages
logger.setLevel(logging.DEBUG)

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
NUCLEAR_PRICE = 9.0  # EUR/MWh - nuclear bid price
NUCLEAR_OPEX = 26.4  # EUR/MWh - nuclear operating cost
OFFSHORE_WIND_OPEX = 24.2  # EUR/MWh - offshore wind operating cost

# Annuity factors for CAPEX calculation (assuming 5% discount rate)
NUCLEAR_LIFETIME = 60  # years
OFFSHORE_WIND_LIFETIME = 25  # years
DISCOUNT_RATE = 0.05
NUCLEAR_CONSTRUCTION_YEARS = 5
OFFSHORE_WIND_CONSTRUCTION_YEARS = 4
# Annuity factor = [1 - (1 + r)^-n] / r
NUCLEAR_ANNUITY_FACTOR = (1 - (1 + DISCOUNT_RATE) ** -NUCLEAR_LIFETIME) / DISCOUNT_RATE
OFFSHORE_WIND_ANNUITY_FACTOR = (1 - (1 + DISCOUNT_RATE) ** -OFFSHORE_WIND_LIFETIME) / DISCOUNT_RATE

AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
OFFSHORE_WIND_AREAS = ["SNII", "UN", "VVD"]  # Offshore wind is in separate busbars
CONNECTED_OFFSHORE_AREAS = {
    "SNII": "NO2",
    "UN": "NO2",
    "VVD": "NO5",
}
MIN_GENERATION_GWH = 1e-6

SCENARIOS = {
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
    # "B": "BASELINE_30TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_NoneNUKE_NoneOFF",
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

TABLE_SCENARIOS = tuple(
    scenario_label for scenario_label in SCENARIOS if not scenario_label.startswith(("SMR", "LMR"))
)


# Setup paths
base_path = Path.cwd()
# Handle running from scripts/paper or workspace root
if base_path.name == "paper":
    base_path = base_path.parent.parent
elif base_path.name == "scripts":
    base_path = base_path.parent

ltm_output_path = base_path / "ltm_output" / MODEL_FOLDER
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER / "paper"
output_path.mkdir(parents=True, exist_ok=True)


def _load_processed_dispatch_data(scenario_path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    data = load_processed_dispatch_data(scenario_path)
    if data is None:
        logger.warning(f"Missing processed_data.parquet dispatch data for {scenario_path.name}")
    return data


def get_nuclear_generation_and_revenue(
    export_df: pd.DataFrame,
    process_df: pd.DataFrame,
    area: str,
    nuclear_price: float,
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Extract nuclear generation and calculate revenue for a specific area.

    Returns:
        df_nuke_generation: DataFrame with nuclear generation (MW)
        total_revenue: Total revenue (EUR)
        capacity_factor: Nuclear capacity factor
        value_factor: Nuclear value factor
        curtailed_generation: Total curtailed generation (GWh)
    """
    if area not in process_df.index.get_level_values("area"):
        return None, 0.0, 0.0, 0.0, 0.0

    df_area = process_df.xs(area, level="area")
    if "nuclear" not in df_area.columns:
        return None, 0.0, 0.0, 0.0, 0.0

    df_nuke_generation = df_area[["nuclear"]]
    df_price = df_area[["market_price"]]

    n_weather_years = len(df_nuke_generation.index.get_level_values("scenario").unique())
    total_revenue = (df_nuke_generation["nuclear"] * df_price["market_price"]).sum() / n_weather_years

    # Approximate capacity factor using max dispatched as capacity proxy
    max_capacity = df_nuke_generation["nuclear"].max()
    capacity_factor = (
        df_nuke_generation["nuclear"].mean() / max_capacity if max_capacity and max_capacity > 0 else 0.0
    )

    achieved_price = (
        (df_price["market_price"] * df_nuke_generation["nuclear"]).sum() / df_nuke_generation["nuclear"].sum()
        if df_nuke_generation["nuclear"].sum() > 0
        else 0.0
    )
    weighted_price = df_price["market_price"].mean()
    value_factor = achieved_price / weighted_price if weighted_price > 0 else 0.0

    # Market spillage is stored as negative production; report curtailment as a positive value.
    curtailed_gwh = 0.0
    if "spillage" in df_area.columns:
        curtailed_gwh = max(0.0, -df_area["spillage"].sum() / n_weather_years / 1000)

    return df_nuke_generation, total_revenue, capacity_factor, value_factor, curtailed_gwh


def get_offshore_wind_generation_and_revenue(
    export_df: pd.DataFrame, process_df: pd.DataFrame, area: str
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float, float]:
    """
    Extract offshore wind generation and calculate revenue for offshore wind areas (SNII, UN, VVD).

    Returns:
        df_wind_net_generation: DataFrame with net offshore wind generation (MW)
        df_wind_gross_generation: DataFrame with gross offshore wind generation (MW)
        total_revenue: Total revenue (EUR)
        capacity_factor: Offshore wind capacity factor
        value_factor: Offshore wind value factor
        curtailed_generation: Total curtailed generation (GWh)
    """
    if area not in export_df.index.get_level_values("area"):
        return None, None, 0.0, 0.0, 0.0, 0.0
    if area not in process_df.index.get_level_values("area"):
        return None, None, 0.0, 0.0, 0.0, 0.0

    df_wind_gross_generation = export_df.xs(area, level="area")[["offshore_wind"]]
    df_area_process = process_df.xs(area, level="area")
    df_price = df_area_process[["market_price"]]

    if "spillage" in df_area_process.columns:
        spillage = df_area_process["spillage"].reindex(df_wind_gross_generation.index, fill_value=0.0)
    else:
        spillage = pd.Series(0.0, index=df_wind_gross_generation.index)

    # Market spillage is negative; delivered offshore wind is gross generation plus spillage.
    df_wind_net_generation = (
        df_wind_gross_generation["offshore_wind"].add(spillage, fill_value=0.0).clip(lower=0.0).to_frame("offshore_wind")
    )

    n_weather_years = len(df_wind_net_generation.index.get_level_values("scenario").unique())
    total_revenue = (df_wind_net_generation["offshore_wind"] * df_price["market_price"]).sum() / n_weather_years

    curtailed_gwh = max(0.0, -spillage.sum() / n_weather_years / 1000)

    max_capacity = df_wind_gross_generation["offshore_wind"].max()
    capacity_factor = (
        df_wind_net_generation["offshore_wind"].mean() / max_capacity if max_capacity and max_capacity > 0 else 0.0
    )

    achieved_price = (
        (df_price["market_price"] * df_wind_net_generation["offshore_wind"]).sum()
        / df_wind_net_generation["offshore_wind"].sum()
        if df_wind_net_generation["offshore_wind"].sum() > 0
        else 0.0
    )
    weighted_price = df_price["market_price"].mean()
    value_factor = achieved_price / weighted_price if weighted_price > 0 else 0.0

    return df_wind_net_generation, df_wind_gross_generation, total_revenue, capacity_factor, value_factor, curtailed_gwh


logger.info("Starting nuclear and offshore wind revenue calculation")
logger.info(f"Analyzing scenarios: {list(SCENARIOS.keys())}")

# Initialize results storage
results = []

def _breakeven_capex_fid_per_kw(
    annual_net_revenue_meur: float,
    capacity_kw: float,
    annuity_factor: float,
    construction_years: int,
    discount_rate: float,
) -> float:
    """
    Break-even CAPEX at construction start (FID), EUR/kW.
    Revenue starts after construction completion.
    """
    if capacity_kw <= 0:
        return 0.0

    pv_revenue_meur_at_fid = (annual_net_revenue_meur * annuity_factor) / ((1 + discount_rate) ** construction_years)
    return (pv_revenue_meur_at_fid * 1e6) / capacity_kw


# Process each scenario
for scenario_label, scenario_name in SCENARIOS.items():
    scenario_path = ltm_output_path / scenario_name

    if not scenario_path.exists():
        logger.warning(f"Scenario path does not exist: {scenario_path}")
        continue

    data = _load_processed_dispatch_data(scenario_path)
    if data is None:
        logger.warning(f"Missing processed dispatch data for {scenario_label}")
        continue
    export_df, process_df = data

    logger.info(f"\nProcessing {scenario_label}: {scenario_name}")

    # Process each area for nuclear
    for area in AREAS:
        logger.info(f"  Processing area: {area}")

        # Get nuclear data if this is a nuclear scenario
        if any(tag in scenario_label for tag in ("N", "SMR", "LMR")):
            try:
                nuke_gen, nuke_revenue, nuke_cf, nuke_vf, nuke_curtailed = get_nuclear_generation_and_revenue(
                    export_df, process_df, area, NUCLEAR_PRICE
                )

                if nuke_gen is not None:
                    n_weather_years = len(nuke_gen.index.get_level_values("scenario").unique())
                    nuke_generation_gwh = nuke_gen.sum().sum() / n_weather_years / 1000  # MW to GWh
                    if nuke_generation_gwh <= MIN_GENERATION_GWH:
                        logger.info(f"    No nuclear in {area}")
                        continue

                    nuke_revenue_meur = nuke_revenue / 1e6  # EUR to MEUR

                    # Calculate operating cost
                    nuke_opex_meur = (nuke_generation_gwh * 1000 * NUCLEAR_OPEX) / 1e6  # MWh * EUR/MWh -> MEUR

                    # Calculate net revenue (revenue - opex)
                    nuke_net_revenue_meur = nuke_revenue_meur - nuke_opex_meur

                    # Calculate installed capacity from generation and capacity factor
                    # Generation (MWh) = Capacity (MW) × 8760 × CF
                    # Capacity (MW) = Generation (MWh) / (8760 × CF)
                    nuke_capacity_mw = (nuke_generation_gwh * 1000) / (8760 * nuke_cf) if nuke_cf > 0 else 0.0

                    # Calculate inferred CAPEX for break-even at construction start (FID), EUR/kW.
                    nuke_capacity_kw = nuke_capacity_mw * 1000
                    nuke_capex_fid = _breakeven_capex_fid_per_kw(
                        annual_net_revenue_meur=nuke_net_revenue_meur,
                        capacity_kw=nuke_capacity_kw,
                        annuity_factor=NUCLEAR_ANNUITY_FACTOR,
                        construction_years=NUCLEAR_CONSTRUCTION_YEARS,
                        discount_rate=DISCOUNT_RATE,
                    )

                    results.append(
                        {
                            "scenario": scenario_label,
                            "area": area,
                            "technology": "Nuclear",
                            "generation_gwh": nuke_generation_gwh,
                            "curtailed_gwh": nuke_curtailed,
                            "revenue_meur": nuke_revenue_meur,
                            "opex_meur": nuke_opex_meur,
                            "net_revenue_meur": nuke_net_revenue_meur,
                            "capacity_factor": nuke_cf,
                            "value_factor": nuke_vf,
                            "revenue_per_mwh": nuke_revenue_meur * 1000 / nuke_generation_gwh
                            if nuke_generation_gwh > 0
                            else 0.0,
                            "breakeven_capex_eur_per_kw": nuke_capex_fid,
                            "construction_years": NUCLEAR_CONSTRUCTION_YEARS,
                        }
                    )

                    logger.info(
                        f"    Nuclear: {nuke_generation_gwh:.2f} GWh (curtailed: {nuke_curtailed:.2f} GWh), {nuke_revenue_meur:.2f} M€, CF={nuke_cf:.3f}, VF={nuke_vf:.3f}"
                    )
                else:
                    logger.info(f"    No nuclear in {area}")

            except Exception as e:
                logger.warning(f"    Failed to process nuclear for {area}: {e}")

    # Process offshore wind areas separately
    for area in OFFSHORE_WIND_AREAS:
        logger.info(f"  Processing offshore wind area: {area}")

        # Get offshore wind data if this is an offshore wind scenario
        if "OW" in scenario_label:
            try:
                (
                    wind_net_gen,
                    wind_gross_gen,
                    wind_revenue,
                    wind_cf,
                    wind_vf,
                    wind_curtailed,
                ) = get_offshore_wind_generation_and_revenue(export_df, process_df, area)

                if wind_net_gen is not None:
                    n_weather_years = len(wind_net_gen.index.get_level_values("scenario").unique())
                    wind_net_generation_gwh = wind_net_gen.sum().sum() / n_weather_years / 1000  # MW to GWh
                    wind_gross_generation_gwh = wind_gross_gen.sum().sum() / n_weather_years / 1000
                    if wind_net_generation_gwh <= MIN_GENERATION_GWH:
                        logger.info(f"    No offshore wind in {area}")
                        continue

                    wind_revenue_meur = wind_revenue / 1e6  # EUR to MEUR

                    # Calculate operating cost
                    wind_opex_meur = (
                        wind_net_generation_gwh * 1000 * OFFSHORE_WIND_OPEX
                    ) / 1e6  # MWh * EUR/MWh -> MEUR

                    # Calculate net revenue (revenue - opex)
                    wind_net_revenue_meur = wind_revenue_meur - wind_opex_meur

                    # Calculate installed capacity from generation and capacity factor
                    # Generation (MWh) = Capacity (MW) × 8760 × CF
                    wind_capacity_mw = (wind_net_generation_gwh * 1000) / (8760 * wind_cf) if wind_cf > 0 else 0.0

                    # Calculate inferred CAPEX for break-even at construction start (FID), EUR/kW.
                    wind_capacity_kw = wind_capacity_mw * 1000
                    wind_capex_fid = _breakeven_capex_fid_per_kw(
                        annual_net_revenue_meur=wind_net_revenue_meur,
                        capacity_kw=wind_capacity_kw,
                        annuity_factor=OFFSHORE_WIND_ANNUITY_FACTOR,
                        construction_years=OFFSHORE_WIND_CONSTRUCTION_YEARS,
                        discount_rate=DISCOUNT_RATE,
                    )

                    results.append(
                        {
                            "scenario": scenario_label,
                            "area": area,
                            "technology": "Offshore Wind",
                            "generation_gwh": wind_net_generation_gwh,
                            "curtailed_gwh": wind_curtailed,
                            "gross_generation_gwh": wind_gross_generation_gwh,
                            "revenue_meur": wind_revenue_meur,
                            "opex_meur": wind_opex_meur,
                            "net_revenue_meur": wind_net_revenue_meur,
                            "capacity_factor": wind_cf,
                            "value_factor": wind_vf,
                            "revenue_per_mwh": wind_revenue_meur * 1000 / wind_net_generation_gwh
                            if wind_net_generation_gwh > 0
                            else 0.0,
                            "breakeven_capex_eur_per_kw": wind_capex_fid,
                            "construction_years": OFFSHORE_WIND_CONSTRUCTION_YEARS,
                        }
                    )

                    logger.info(
                        f"    Offshore Wind: {wind_net_generation_gwh:.2f} GWh (curtailed: {wind_curtailed:.2f} GWh), {wind_revenue_meur:.2f} M€, CF={wind_cf:.3f}, VF={wind_vf:.3f}"
                    )
                else:
                    logger.info(f"    No offshore wind in {area}")

            except Exception as e:
                logger.warning(f"    Failed to process offshore wind for {area}: {e}")

# Create results DataFrame
df_results = pd.DataFrame(results)

if df_results.empty:
    logger.error("No results generated!")
    raise ValueError("No results generated!")

# Convert GWh to TWh
df_results["generation_twh"] = df_results["generation_gwh"] / 1000

# Save to CSV
output_csv = output_path / "nuclear_offshore_revenue.csv"
df_results.to_csv(output_csv, index=False)
logger.info(f"\nSaved results to {output_csv}")

# Print summary
logger.info("\n" + "=" * 80)
logger.info("SUMMARY BY SCENARIO AND TECHNOLOGY")
logger.info("=" * 80)

summary = (
    df_results.groupby(["scenario", "technology"])
    .agg(
        {
            "generation_twh": "sum",
            "revenue_meur": "sum",
            "capacity_factor": "mean",
            "value_factor": "mean",
        }
    )
    .round(2)
)

print(summary)

# Create LaTeX table
latex_output = output_path / "nuclear_offshore_revenue.tex"

# Prepare data for LaTeX table with areas
latex_rows = []
# Get unique scenarios from results, maintaining order from SCENARIOS dict
scenarios_in_results = [s for s in TABLE_SCENARIOS if s in df_results["scenario"].unique()]

for i, scenario in enumerate(scenarios_in_results):
    scenario_data = df_results[df_results["scenario"] == scenario]

    # Collect all rows for this scenario
    scenario_rows = []

    # Nuclear rows by area
    nuclear_data = scenario_data[scenario_data["technology"] == "Nuclear"].sort_values("area")
    for _, row in nuclear_data.iterrows():
        if row["generation_gwh"] <= MIN_GENERATION_GWH:
            continue
        gen = row["generation_twh"]
        rev_per_mwh = row["revenue_per_mwh"]
        curtail = row["curtailed_gwh"] / 1000  # Convert to TWh
        cf = row["capacity_factor"]
        vf = row["value_factor"]
        capex = row["breakeven_capex_eur_per_kw"]
        area = row["area"]
        scenario_rows.append(
            f"{area} & Nuclear & {gen:.1f} & {rev_per_mwh:.1f} & {curtail:.2f} & {cf:.2f} & {vf:.2f} & {capex:.0f}"
        )

    # Offshore Wind rows by area
    wind_data = scenario_data[scenario_data["technology"] == "Offshore Wind"].sort_values("area")
    for _, row in wind_data.iterrows():
        if row["generation_gwh"] <= MIN_GENERATION_GWH:
            continue
        gen = row["generation_twh"]
        rev_per_mwh = row["revenue_per_mwh"]
        curtail = row["curtailed_gwh"] / 1000  # Convert to TWh
        cf = row["capacity_factor"]
        vf = row["value_factor"]
        capex = row["breakeven_capex_eur_per_kw"]
        area = row["area"]
        scenario_rows.append(
            f"{area} & Offshore Wind & {gen:.1f} & {rev_per_mwh:.1f} & {curtail:.2f} & {cf:.2f} & {vf:.2f} & {capex:.0f}"
        )

    # Add multirow for scenario
    if scenario_rows:
        n_rows = len(scenario_rows)
        # First row with multirow - format scenario name with \texttt{}
        scenario_formatted = f"\\texttt{{{scenario}}}"
        latex_rows.append(f"\\multirow{{{n_rows}}}{{*}}{{{scenario_formatted}}} & {scenario_rows[0]} \\\\")
        # Remaining rows without scenario name
        for row in scenario_rows[1:]:
            latex_rows.append(f" & {row} \\\\")
        # Add midrule after each scenario except the last
        if i < len(scenarios_in_results) - 1:
            latex_rows.append("\\midrule")

# Write LaTeX table
latex_table = (
    r"""\begin{table}[htbp]
\centering
\caption{Nuclear and offshore wind generation, revenue, and performance factors by area. Gen. is annual delivered generation averaged over weather years. Rev. is generation-weighted market revenue. Area spill. is local market spillage in the listed area, reported as a positive annual energy volume; it is not technology-specific in mixed price areas and only covers local spillage for offshore wind busbars. Cap. Factor is mean delivered generation divided by inferred maximum capacity. Value Factor is achieved generation-weighted price divided by the area mean price. Inferred CAPEX is the break-even investment cost at FID after operating costs.}
\label{tab:nuclear_offshore_revenue}
\begin{tabular}{lllrrrrrrr}
\toprule
Scenario & Area & Tech. & Gen. & Rev. & Area spill. & Cap. & Value & Inferred CAPEX \\
         &      &       & (TWh) & (EUR/MWh) & (TWh) & Factor & Factor & for break-even (EUR/kW) \\
\midrule
"""
    + "\n".join(latex_rows)
    + r"""
\bottomrule
\end{tabular}
\end{table}"""
)

with open(latex_output, "w") as f:
    f.write(latex_table)

print(latex_table)

logger.info(f"\nSaved LaTeX table to {latex_output}")

logger.info(f"\nAnalysis complete. Results saved to {output_path}")
