"""
Calculate curtailment for all scenarios and areas.

This script:
1. Loads specified scenarios (BA and LLPS with N, OW, OWN variants)
2. For each scenario, calculates curtailment in each area (negative market steps)
3. Computes total curtailment for the entire system
4. Computes total curtailment for Norway (NO1-NO5)
5. Outputs results to CSV and LaTeX table
"""

import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging

import pandas as pd

from scripts.common import load_scenarios, logger

# Set logger level to INFO
logger.setLevel(logging.INFO)

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Scenarios to analyze
SCENARIOS = [
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
]

# Shorter names for display
SCENARIO_LABELS = {
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-LLPS+",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-LLPS+",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-LLPS+",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA+",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-BA+",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA+",
    "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-LLPS",
    "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-LLPS",
    "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-LLPS",
    "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "N-BA",
    "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "OWN-BA",
    "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "OW-BA",
}

# Norwegian areas to aggregate
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# Setup paths
base_path = Path.cwd()
# Handle running from scripts/paper or workspace root
if base_path.name == "paper":
    base_path = base_path.parent.parent
elif base_path.name == "scripts":
    base_path = base_path.parent

output_path = base_path / OUTPUT_DIR / MODEL_FOLDER / "paper"
output_path.mkdir(parents=True, exist_ok=True)


def get_curtailment_by_area(scenario_data) -> dict:
    """
    Calculate curtailment for all areas in a scenario.

    Curtailment is identified as negative production from market steps.

    Returns:
        Dictionary mapping area names to curtailment in GWh
    """
    curtailment_by_area = {}

    # Process each area
    for area in scenario_data.get_busbar_names():
        # Get market step production for this area
        df_market_steps = scenario_data.get_market_steps_for_busbar(area)

        # Curtailment is negative production (use < 0.01 to avoid numerical noise)
        df_curtailment = df_market_steps[df_market_steps < 0.01]

        # Take absolute value and sum
        total_curtailment = df_curtailment.abs().sum().sum()

        if total_curtailment > 0:
            # Average over weather years
            n_weather_years = df_market_steps.shape[1]
            curtailment_mwh = total_curtailment / n_weather_years
            curtailment_gwh = curtailment_mwh / 1000
            curtailment_by_area[area] = curtailment_gwh

    return curtailment_by_area

    return curtailment_by_area


logger.info("Starting curtailment calculation")
logger.info(f"Analyzing {len(SCENARIOS)} scenarios")

# Load scenarios
scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

logger.info(f"Loaded {len(scenarios)} scenarios")

# Initialize results storage
results = []

# Process each scenario
for scenario_name, scenario_data in scenarios.items():
    scenario_label = SCENARIO_LABELS.get(scenario_name, scenario_name)
    logger.info(f"\nProcessing: {scenario_label}")

    try:
        # Get curtailment by area
        curtailment_by_area = get_curtailment_by_area(scenario_data)

        if not curtailment_by_area:
            logger.info("  No curtailment detected")
            # Still add a row with zeros
            results.append(
                {
                    "scenario": scenario_label,
                    "total_system": 0.0,
                    "total_norway": 0.0,
                    **{area: 0.0 for area in NO_AREAS},
                }
            )
            continue

        # Calculate totals
        total_system = sum(curtailment_by_area.values())
        total_norway = sum(curtailment_by_area.get(area, 0.0) for area in NO_AREAS)

        logger.info(f"  Total system curtailment: {total_system:.2f} GWh")
        logger.info(f"  Total Norway curtailment: {total_norway:.2f} GWh")

        # Add to results
        result_row = {
            "scenario": scenario_label,
            "total_system": total_system,
            "total_norway": total_norway,
        }

        # Add individual areas
        for area in NO_AREAS:
            result_row[area] = curtailment_by_area.get(area, 0.0)

        # Add any other areas (like SNII, UN, VVD)
        for area, curtailment in curtailment_by_area.items():
            if area not in NO_AREAS and area not in result_row:
                result_row[area] = curtailment

        results.append(result_row)

        # Log area-specific curtailment
        for area, curtailment in sorted(curtailment_by_area.items()):
            if curtailment > 0:
                logger.info(f"    {area}: {curtailment:.2f} GWh")

    except Exception as e:
        logger.error(f"  Failed to process scenario: {e}")
        import traceback

        traceback.print_exc()

# Create results DataFrame
df_results = pd.DataFrame(results)

if df_results.empty:
    logger.error("No results generated!")
    raise ValueError("No results generated!")

# Fill NaN with 0
df_results = df_results.fillna(0.0)

# Convert GWh to TWh for display
df_display = df_results.copy()
for col in df_display.columns:
    if col != "scenario":
        df_display[col] = df_display[col] / 1000

# Save to CSV
output_csv = output_path / "curtailment_all_scenarios.csv"
df_display.to_csv(output_csv, index=False)
logger.info(f"\nSaved results to {output_csv}")

# Print summary statistics
logger.info("\n" + "=" * 80)
logger.info("SUMMARY STATISTICS")
logger.info("=" * 80)

logger.info("\nTop 10 scenarios by total system curtailment:")
top_scenarios = df_display.nlargest(10, "total_system")[["scenario", "total_system", "total_norway"]]
print(top_scenarios.to_string(index=False))

logger.info("\nTop 10 scenarios by Norway curtailment:")
top_norway = df_display.nlargest(10, "total_norway")[["scenario", "total_system", "total_norway"]]
print(top_norway.to_string(index=False))

# Create LaTeX table for scenarios with significant curtailment (> 0.1 TWh)
significant_curtailment = df_display[df_display["total_system"] > 0.1].sort_values("total_system", ascending=False)

if not significant_curtailment.empty:
    latex_output = output_path / "curtailment_summary.tex"

    latex_rows = []
    for _, row in significant_curtailment.head(20).iterrows():  # Top 20 scenarios
        scenario = row["scenario"]
        # Abbreviate scenario name for table
        scenario_short = scenario.split("_")[0] + "_" + scenario.split("_")[1] if "_" in scenario else scenario
        total_sys = row["total_system"]
        total_no = row["total_norway"]
        no1 = row.get("NO1", 0)
        no2 = row.get("NO2", 0)
        no3 = row.get("NO3", 0)
        no4 = row.get("NO4", 0)
        no5 = row.get("NO5", 0)

        latex_rows.append(
            f"{scenario_short} & {total_sys:.2f} & {total_no:.2f} & {no1:.2f} & {no2:.2f} & {no3:.2f} & {no4:.2f} & {no5:.2f} \\\\"
        )

    latex_table = (
        r"""\begin{table}[htbp]
\centering
\caption{Curtailment by Scenario and Area (TWh)}
\label{tab:curtailment_summary}
\begin{tabular}{lrrrrrrr}
\toprule
Scenario & Total & Norway & NO1 & NO2 & NO3 & NO4 & NO5 \\
         & System & Total &     &     &     &     &     \\
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

    logger.info(f"\nSaved LaTeX table to {latex_output}")
else:
    logger.info("\nNo scenarios with significant curtailment (> 0.1 TWh)")

logger.info(f"\nAnalysis complete. Results saved to {output_path}")


print(latex_table)
