"""
Compute reservoir spillage across all areas for OW_N_OWN scenarios.
Aggregates spillage at the busbar level and generates summary statistics.
"""

from pathlib import Path

import pandas as pd
from scripts.common import df_from_pyltm_result

from scripts.common import load_scenarios, logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Scenarios from OW_N_OWN group
SCENARIOS = [
    "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
]

# Shorter names for display
SCENARIO_LABELS = {
    "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF": "BASELINE",
    "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "LLPS_N",
    "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "LLPS_OWN",
    "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "LLPS_OW",
    "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF": "BA_N",
    "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF": "BA_OWN",
    "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF": "BA_OW",
}

# Norwegian areas to aggregate
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# Setup paths
base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)

# Load scenarios
scenario_paths = {name: base_path / f"ltm_output/{MODEL_FOLDER}/{name}" for name in SCENARIOS}
scenarios = load_scenarios(scenario_paths)

if not scenarios:
    logger.error("No scenarios loaded")
    exit(1)

logger.info(f"Loaded {len(scenarios)} scenarios")


# Collect spillage data for each scenario and area
spillage_results = {}
spillage_statistics = []

for scenario_name, scenario in scenarios.items():
    short_name = SCENARIO_LABELS.get(scenario_name, scenario_name)
    logger.info(f"\nProcessing scenario: {short_name}")

    scenario_spillage = {}

    for area in NO_AREAS:
        try:
            busbars_dict = scenario.get_busbars()
            busbar = busbars_dict[area]

            # Aggregate spillage and discharge from all reservoirs in this area
            total_spillage = None
            total_discharge = None
            reservoir_count = 0

            for rsv in busbar.reservoirs():
                try:
                    df_spill = df_from_pyltm_result(rsv.spill(time_axis=True))
                    df_discharge = df_from_pyltm_result(rsv.discharge(time_axis=True))

                    if total_spillage is None:
                        total_spillage = df_spill
                        total_discharge = df_discharge
                    else:
                        total_spillage = total_spillage + df_spill
                        total_discharge = total_discharge + df_discharge

                    reservoir_count += 1
                except Exception as e:
                    logger.warning(f"Failed to get data for reservoir {rsv.name} in {area}: {e}")

            if total_spillage is not None and total_discharge is not None:
                # Calculate mean flow rates (m³/s)
                mean_spillage = total_spillage.mean().mean()
                mean_discharge = total_discharge.mean().mean()

                # Calculate spillage as percentage of discharge
                spillage_percentage = (mean_spillage / mean_discharge * 100) if mean_discharge > 0 else 0

                # Store the full DataFrame
                scenario_spillage[area] = {
                    "mean_spillage_m3s": mean_spillage,
                    "mean_discharge_m3s": mean_discharge,
                    "spillage_percentage": spillage_percentage,
                    "reservoirs": reservoir_count,
                }

                # Add to statistics list
                spillage_statistics.append(
                    {
                        "Scenario": short_name,
                        "Area": area,
                        "Mean Spillage (m³/s)": mean_spillage,
                        "Mean Discharge (m³/s)": mean_discharge,
                        "Spillage/Discharge (%)": spillage_percentage,
                        "Num Reservoirs": reservoir_count,
                    }
                )

                logger.info(
                    f"  {area}: Spillage={mean_spillage:.2f} m³/s, Discharge={mean_discharge:.2f} m³/s, "
                    f"Spillage%={spillage_percentage:.2f}% ({reservoir_count} reservoirs)"
                )
            else:
                logger.warning(f"  {area}: No spillage/discharge data available")

        except Exception as e:
            logger.error(f"Failed to process {area} in {scenario_name}: {e}")

    spillage_results[short_name] = scenario_spillage

# Create summary DataFrame
df_statistics = pd.DataFrame(spillage_statistics)

# Create pivot tables for easier comparison
df_spillage_pct = df_statistics.pivot(index="Area", columns="Scenario", values="Spillage/Discharge (%)")
df_mean_spillage = df_statistics.pivot(index="Area", columns="Scenario", values="Mean Spillage (m³/s)")
df_mean_discharge = df_statistics.pivot(index="Area", columns="Scenario", values="Mean Discharge (m³/s)")

# Add summary row
df_spillage_pct.loc["MEAN"] = df_spillage_pct.mean()
df_mean_spillage.loc["TOTAL"] = df_mean_spillage.sum()
df_mean_discharge.loc["TOTAL"] = df_mean_discharge.sum()

# Calculate spillage as percentage of BASELINE
if "BASELINE" in df_mean_spillage.columns:
    baseline_spillage = df_mean_spillage["BASELINE"]

    # Create percentage change relative to BASELINE
    df_spillage_pct_of_baseline = pd.DataFrame(index=df_mean_spillage.index)

    for col in df_mean_spillage.columns:
        if col != "BASELINE":
            # Calculate percentage: (scenario - baseline) / baseline * 100
            df_spillage_pct_of_baseline[col] = (df_mean_spillage[col] - baseline_spillage) / baseline_spillage * 100

    # BASELINE itself is 0% (no change from itself)
    df_spillage_pct_of_baseline["BASELINE"] = 0.0

    # Reorder columns to have BASELINE first
    cols = ["BASELINE"] + [c for c in df_spillage_pct_of_baseline.columns if c != "BASELINE"]
    df_spillage_pct_of_baseline = df_spillage_pct_of_baseline[cols]

    # Calculate absolute change from BASELINE
    df_spillage_abs_change = pd.DataFrame(index=df_mean_spillage.index)

    for col in df_mean_spillage.columns:
        if col != "BASELINE":
            # Calculate absolute difference: scenario - baseline
            df_spillage_abs_change[col] = df_mean_spillage[col] - baseline_spillage

    # BASELINE itself is 0.0 (no change from itself)
    df_spillage_abs_change["BASELINE"] = 0.0

    # Reorder columns to have BASELINE first
    df_spillage_abs_change = df_spillage_abs_change[cols]

    logger.info("\nCalculated spillage as percentage change from BASELINE")
else:
    logger.warning("BASELINE scenario not found, skipping percentage of baseline calculation")
    df_spillage_pct_of_baseline = None
    df_spillage_abs_change = None

# Save to CSV files
output_file_percentage = paper_output_path / "spillage_percentage_by_area.csv"
output_file_spillage = paper_output_path / "spillage_mean_by_area.csv"
output_file_discharge = paper_output_path / "discharge_mean_by_area.csv"
output_file_all = paper_output_path / "spillage_all_statistics.csv"
output_file_pct_baseline = paper_output_path / "spillage_pct_of_baseline.csv"
output_file_abs_change = paper_output_path / "spillage_abs_change_from_baseline.csv"

df_spillage_pct.to_csv(output_file_percentage)
df_mean_spillage.to_csv(output_file_spillage)
df_mean_discharge.to_csv(output_file_discharge)
df_statistics.to_csv(output_file_all, index=False)

if df_spillage_pct_of_baseline is not None:
    df_spillage_pct_of_baseline.to_csv(output_file_pct_baseline)
if df_spillage_abs_change is not None:
    df_spillage_abs_change.to_csv(output_file_abs_change)

logger.info("\nSaved spillage statistics to:")
logger.info(f"  Percentage: {output_file_percentage}")
logger.info(f"  Spillage: {output_file_spillage}")
logger.info(f"  Discharge: {output_file_discharge}")
logger.info(f"  All: {output_file_all}")
if df_spillage_pct_of_baseline is not None:
    logger.info(f"  % of BASELINE: {output_file_pct_baseline}")
if df_spillage_abs_change is not None:
    logger.info(f"  Abs change: {output_file_abs_change}")


# Helper function to create structured LaTeX tables
def create_structured_latex_table(df, caption, label, value_format=".2f", show_sign=False):
    """Create LaTeX table with BA/LLPS main categories and N/OW/OWN subcategories."""

    # Define column structure: BA (N, OWN, OW), then LLPS (N, OWN, OW) - exclude BASELINE
    col_structure = {
        "BA": ["BA_N", "BA_OWN", "BA_OW"],
        "LLPS": ["LLPS_N", "LLPS_OWN", "LLPS_OW"],
    }

    # Check which columns exist
    available_cols = []
    col_groups = []
    for group, cols in col_structure.items():
        available = [c for c in cols if c in df.columns]
        if available:
            col_groups.append((group, available))
            available_cols.extend(available)

    if not available_cols:
        return None

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")

    # Build column specification
    n_cols = len(available_cols)
    lines.append("\\begin{tabular}{l" + "r" * n_cols + "}")
    lines.append("\\toprule")

    # Top header row with main categories
    top_header = "\\textbf{Area}"
    for group, cols in col_groups:
        top_header += " & \\multicolumn{" + str(len(cols)) + "}{c}{\\textbf{" + group + "}}"
    top_header += " \\\\"
    lines.append(top_header)

    # Add cmidrule under each main category
    cmidrules = []
    col_start = 2  # Start after the Area column
    for group, cols in col_groups:
        col_end = col_start + len(cols) - 1
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1
    lines.append("".join(cmidrules))

    # Second header row with subcategories
    sub_header = "\\textbf{Area}"
    for group, cols in col_groups:
        for col in cols:
            # Extract subcategory (N, OW, OWN)
            subcategory = col.split("_", 1)[1]  # e.g., "BA_N" -> "N"
            sub_header += " & \\textbf{" + subcategory + "}"
    sub_header += " \\\\"
    lines.append(sub_header)
    lines.append("\\midrule")

    # Data rows
    for area in df.index:
        row_values = [area]
        for col in available_cols:
            val = df.loc[area, col]
            if pd.isna(val):
                row_values.append("-")
            elif val == 0.0 and show_sign:
                row_values.append("0.00")
            elif show_sign:
                row_values.append(f"{val:+{value_format}}")
            else:
                row_values.append(f"{val:{value_format}}")
        lines.append(" & ".join(row_values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# Generate LaTeX tables with restructured format
latex_table_pct = create_structured_latex_table(
    df_spillage_pct, "Spillage as Percentage of Discharge by Area (\\%)", "tab:spillage_percentage", value_format=".2f"
)

latex_table_pct_baseline = None
latex_table_abs_change = None

if df_spillage_pct_of_baseline is not None:
    latex_table_pct_baseline = create_structured_latex_table(
        df_spillage_pct_of_baseline,
        "Spillage Change Relative to BASELINE (\\%)",
        "tab:spillage_pct_baseline",
        value_format=".1f",
        show_sign=True,
    )

if df_spillage_abs_change is not None:
    latex_table_abs_change = create_structured_latex_table(
        df_spillage_abs_change,
        "Absolute Spillage Change from BASELINE (m³/s)",
        "tab:spillage_abs_change",
        value_format=".1f",
        show_sign=True,
    )

# Write LaTeX tables to files
if latex_table_pct:
    output_file_latex = paper_output_path / "spillage_percentage_table.tex"
    with open(output_file_latex, "w") as f:
        f.write(latex_table_pct)
    logger.info(f"  LaTeX: {output_file_latex}")

if latex_table_pct_baseline:
    output_file_latex_baseline = paper_output_path / "spillage_pct_of_baseline_table.tex"
    with open(output_file_latex_baseline, "w") as f:
        f.write(latex_table_pct_baseline)
    logger.info(f"  LaTeX (% of BASELINE): {output_file_latex_baseline}")

if latex_table_abs_change:
    output_file_latex_abs_change = paper_output_path / "spillage_abs_change_table.tex"
    with open(output_file_latex_abs_change, "w") as f:
        f.write(latex_table_abs_change)
    logger.info(f"  LaTeX (abs change): {output_file_latex_abs_change}")

# Print summary to console
print("\n" + "=" * 80)
print("SPILLAGE SUMMARY - Spillage as % of Discharge by Area")
print("=" * 80)
print(df_spillage_pct.to_string())

print("\n" + "=" * 80)
print("SPILLAGE SUMMARY - Mean Spillage by Area (m³/s)")
print("=" * 80)
print(df_mean_spillage.to_string())

print("\n" + "=" * 80)
print("SPILLAGE SUMMARY - Mean Discharge by Area (m³/s)")
print("=" * 80)
print(df_mean_discharge.to_string())

if df_spillage_pct_of_baseline is not None:
    print("\n" + "=" * 80)
    print("SPILLAGE SUMMARY - Change Relative to BASELINE (%)")
    print("=" * 80)
    print(df_spillage_pct_of_baseline.to_string())

if df_spillage_abs_change is not None:
    print("\n" + "=" * 80)
    print("SPILLAGE SUMMARY - Absolute Change from BASELINE (m³/s)")
    print("=" * 80)
    print(df_spillage_abs_change.to_string())

print("\n" + "=" * 80)
print(f"Results saved to: {paper_output_path}")
print("=" * 80)


# %%


def delme():
    
    scenario = scenarios["LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF"]
    busbars_dict = scenario.get_busbars()
    
    busbar = busbars_dict["NO2"]
    reservoirs = {rsv.name: rsv for rsv in busbar.reservoirs()}
    rsv = reservoirs["res_osen"]
    df_spill = df_from_pyltm_result(rsv.spill(time_axis=True))
    df_spill.sum().sum()
    df_spill.mean().plot()

    df_dis = df_from_pyltm_result(rsv.discharge(time_axis=True))
    df_dis.mean().plot()

    df_prod = df_from_pyltm_result(rsv.production(time_axis=True))
    df_res = df_from_pyltm_result(rsv.reservoir(time_axis=True))
    df_res.plot()

    df_res.max().max()

    rsv.max_volume
    vals = {"spill": 0.0, "dis": 0.0}
    for r, rsv in reservoirs.items():
        df_spill = df_from_pyltm_result(rsv.spill(time_axis=True))
        df_prod = df_from_pyltm_result(rsv.production(time_axis=True))
        df_dis = df_from_pyltm_result(rsv.discharge(time_axis=True))
        df_res = df_from_pyltm_result(rsv.reservoir(time_axis=True))
        print(
            f"{r}: {df_spill.mean().mean():.2f} m3/s, {df_dis.mean().mean():.2f} m3/s, {df_prod.mean().mean():.2f} MW, {df_res.max().max():.2f} Mm3"
        )
        vals["spill"] += df_spill.mean().mean()
        vals["dis"] += df_dis.mean().mean()
    # %%

    vals["spill"] / vals["dis"]

    # %%

    df = df_from_pyltm_result(busbar.sum_reservoir())
    df.plot()
    df = df_from_pyltm_result(busbar.sum_hydro_production())
    df.mean(axis=1).plot()

    df

    df_load_1 = df_from_pyltm_result(b1.sum_load())
    df_load_2 = df_from_pyltm_result(b2.sum_load())
