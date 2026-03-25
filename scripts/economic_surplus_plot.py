"""
Plot and export tables for economic surplus results.
Reads processed data from calculate_economic_surplus.py output.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import logger

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

base_path = Path.cwd()
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)
processed_output_path = paper_output_path / "economic_surplus_data.pkl"

if not processed_output_path.exists():
    raise FileNotFoundError(f"Processed surplus data not found: {processed_output_path}")

payload = pd.read_pickle(processed_output_path)
surplus_results = payload["surplus_results"]
technology_surplus_results = payload["technology_surplus_results"]
SCENARIO_LABELS = payload["scenario_labels"]
SCENARIO_GROUPS = payload["scenario_groups"]
ALL_AREAS = payload["all_areas"]
NO_AREAS = payload["no_areas"]
OFFSHORE_WIND_AREAS = payload["offshore_areas"]

# Scenario ordering and baseline derived from process payload
scenario_order = [short for short in SCENARIO_LABELS.values() if short in surplus_results]
if not scenario_order:
    scenario_order = list(surplus_results.keys())

baseline_short = next(
    (short for full, short in SCENARIO_LABELS.items() if "BASELINE" in full and short in surplus_results),
    "BASELINE",
)


def _detect_load_profile(short_name: str) -> str:
    if "BA" in short_name:
        return "BA"
    if "LLPS" in short_name:
        return "LLPS"
    return "Other"


def _detect_generation_type(short_name: str) -> str:
    if "OWN" in short_name:
        return "OWN"
    if "OW" in short_name:
        return "OW"
    if "N" in short_name:
        return "N"
    return short_name


def _display_scenario_name(scenario_name: str) -> str:
    return "B" if scenario_name == baseline_short else scenario_name

# ============================================================================
# Create visualization comparing surplus across scenarios
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Define colors
COLORS = {
    "consumer": "#1f77b4",  # Blue
    "producer": "#ff7f0e",  # Orange
    "societal": "#2ca02c",  # Green
}

# Plot 1: Norway Total Surplus by Scenario Type
ax = axes[0, 0]
x_groups = np.arange(3)  # N, OWN, OW
bar_width = 0.25

norway_cs = []
norway_ps = []
norway_ss = []

for group_name in ["N", "OWN", "OW"]:
    scenarios_in_group = SCENARIO_GROUPS[group_name]
    # Average LLPS and BA for each group
    cs_values = [
        surplus_results.get(s, {}).get("norway_total", {}).get("consumer_surplus", 0)
        for s in scenarios_in_group
        if s in surplus_results
    ]
    ps_values = [
        surplus_results.get(s, {}).get("norway_total", {}).get("producer_surplus", 0)
        for s in scenarios_in_group
        if s in surplus_results
    ]
    ss_values = [
        surplus_results.get(s, {}).get("norway_total", {}).get("societal_surplus", 0)
        for s in scenarios_in_group
        if s in surplus_results
    ]

    norway_cs.append(np.mean(cs_values) if cs_values else 0)
    norway_ps.append(np.mean(ps_values) if ps_values else 0)
    norway_ss.append(np.mean(ss_values) if ss_values else 0)

ax.bar(x_groups - bar_width, norway_cs, bar_width, label="Consumer Surplus", color=COLORS["consumer"], alpha=0.8)
ax.bar(x_groups, norway_ps, bar_width, label="Producer Surplus", color=COLORS["producer"], alpha=0.8)
ax.bar(x_groups + bar_width, norway_ss, bar_width, label="Societal Surplus", color=COLORS["societal"], alpha=0.8)

ax.set_xlabel("Scenario Type", fontsize=12)
ax.set_ylabel("Surplus (M€)", fontsize=12)
ax.set_title("Economic Surplus - Norway Total", fontsize=14, fontweight="bold")
ax.set_xticks(x_groups)
ax.set_xticklabels(["N (Nuclear)", "OWN (Offshore + Nuclear)", "OW (Offshore)"])
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

# Plot 2: All Nordic Areas Total Surplus by Scenario Type
ax = axes[0, 1]

all_cs = []
all_ps = []
all_ss = []

for group_name in ["N", "OWN", "OW"]:
    scenarios_in_group = SCENARIO_GROUPS[group_name]
    cs_values = [
        surplus_results.get(s, {}).get("all_areas_total", {}).get("consumer_surplus", 0)
        for s in scenarios_in_group
        if s in surplus_results
    ]
    ps_values = [
        surplus_results.get(s, {}).get("all_areas_total", {}).get("producer_surplus", 0)
        for s in scenarios_in_group
        if s in surplus_results
    ]
    ss_values = [
        surplus_results.get(s, {}).get("all_areas_total", {}).get("societal_surplus", 0)
        for s in scenarios_in_group
        if s in surplus_results
    ]

    all_cs.append(np.mean(cs_values) if cs_values else 0)
    all_ps.append(np.mean(ps_values) if ps_values else 0)
    all_ss.append(np.mean(ss_values) if ss_values else 0)

ax.bar(x_groups - bar_width, all_cs, bar_width, label="Consumer Surplus", color=COLORS["consumer"], alpha=0.8)
ax.bar(x_groups, all_ps, bar_width, label="Producer Surplus", color=COLORS["producer"], alpha=0.8)
ax.bar(x_groups + bar_width, all_ss, bar_width, label="Societal Surplus", color=COLORS["societal"], alpha=0.8)

ax.set_xlabel("Scenario Type", fontsize=12)
ax.set_ylabel("Surplus (M€)", fontsize=12)
ax.set_title("Economic Surplus - All Nordic Areas", fontsize=14, fontweight="bold")
ax.set_xticks(x_groups)
ax.set_xticklabels(["N (Nuclear)", "OWN (Offshore + Nuclear)", "OW (Offshore)"])
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

# Plot 3: Change from Baseline - Norway
ax = axes[1, 0]

if baseline_short in surplus_results:
    baseline_no = surplus_results[baseline_short]["norway_total"]

    delta_cs = []
    delta_ps = []
    delta_ss = []

    for group_name in ["N", "OWN", "OW"]:
        scenarios_in_group = SCENARIO_GROUPS[group_name]
        cs_deltas = []
        ps_deltas = []
        ss_deltas = []

        for s in scenarios_in_group:
            if s in surplus_results:
                no_total = surplus_results[s]["norway_total"]
                cs_deltas.append(no_total["consumer_surplus"] - baseline_no["consumer_surplus"])
                ps_deltas.append(no_total["producer_surplus"] - baseline_no["producer_surplus"])
                ss_deltas.append(no_total["societal_surplus"] - baseline_no["societal_surplus"])

        delta_cs.append(np.mean(cs_deltas) if cs_deltas else 0)
        delta_ps.append(np.mean(ps_deltas) if ps_deltas else 0)
        delta_ss.append(np.mean(ss_deltas) if ss_deltas else 0)

    ax.bar(x_groups - bar_width, delta_cs, bar_width, label="Δ Consumer Surplus", color=COLORS["consumer"], alpha=0.8)
    ax.bar(x_groups, delta_ps, bar_width, label="Δ Producer Surplus", color=COLORS["producer"], alpha=0.8)
    ax.bar(x_groups + bar_width, delta_ss, bar_width, label="Δ Societal Surplus", color=COLORS["societal"], alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Scenario Type", fontsize=12)
    ax.set_ylabel("Change from Baseline (M€)", fontsize=12)
    ax.set_title("Surplus Change from Baseline - Norway", fontsize=14, fontweight="bold")
    ax.set_xticks(x_groups)
    ax.set_xticklabels(["N (Nuclear)", "OWN (Offshore + Nuclear)", "OW (Offshore)"])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

# Plot 4: Change from Baseline - All Nordic Areas
ax = axes[1, 1]

if baseline_short in surplus_results:
    baseline_all = surplus_results[baseline_short]["all_areas_total"]

    delta_cs_all = []
    delta_ps_all = []
    delta_ss_all = []

    for group_name in ["N", "OWN", "OW"]:
        scenarios_in_group = SCENARIO_GROUPS[group_name]
        cs_deltas = []
        ps_deltas = []
        ss_deltas = []

        for s in scenarios_in_group:
            if s in surplus_results:
                all_total = surplus_results[s]["all_areas_total"]
                cs_deltas.append(all_total["consumer_surplus"] - baseline_all["consumer_surplus"])
                ps_deltas.append(all_total["producer_surplus"] - baseline_all["producer_surplus"])
                ss_deltas.append(all_total["societal_surplus"] - baseline_all["societal_surplus"])

        delta_cs_all.append(np.mean(cs_deltas) if cs_deltas else 0)
        delta_ps_all.append(np.mean(ps_deltas) if ps_deltas else 0)
        delta_ss_all.append(np.mean(ss_deltas) if ss_deltas else 0)

    ax.bar(
        x_groups - bar_width, delta_cs_all, bar_width, label="Δ Consumer Surplus", color=COLORS["consumer"], alpha=0.8
    )
    ax.bar(x_groups, delta_ps_all, bar_width, label="Δ Producer Surplus", color=COLORS["producer"], alpha=0.8)
    ax.bar(
        x_groups + bar_width, delta_ss_all, bar_width, label="Δ Societal Surplus", color=COLORS["societal"], alpha=0.8
    )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Scenario Type", fontsize=12)
    ax.set_ylabel("Change from Baseline (M€)", fontsize=12)
    ax.set_title("Surplus Change from Baseline - All Nordic Areas", fontsize=14, fontweight="bold")
    ax.set_xticks(x_groups)
    ax.set_xticklabels(["N (Nuclear)", "OWN (Offshore + Nuclear)", "OW (Offshore)"])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

# Save figure
output_file = paper_output_path / "economic_surplus_comparison.pdf"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
logger.info(f"\nSaved surplus comparison plot to: {output_file}")

# ============================================================================
# Create detailed summary tables
# ============================================================================

# Table 1: Absolute surplus values by scenario
summary_data = []
for scenario_name in scenario_order:
    if scenario_name in surplus_results:
        no_total = surplus_results[scenario_name]["norway_total"]
        all_total = surplus_results[scenario_name]["all_areas_total"]

        summary_data.append(
            {
                "Scenario": _display_scenario_name(scenario_name),
                "Region": "Norway",
                "Consumer Surplus (M€)": no_total["consumer_surplus"],
                "Producer Surplus (M€)": no_total["producer_surplus"],
                "Societal Surplus (M€)": no_total["societal_surplus"],
            }
        )

        summary_data.append(
            {
                "Scenario": _display_scenario_name(scenario_name),
                "Region": "All Nordic",
                "Consumer Surplus (M€)": all_total["consumer_surplus"],
                "Producer Surplus (M€)": all_total["producer_surplus"],
                "Societal Surplus (M€)": all_total["societal_surplus"],
            }
        )

df_summary = pd.DataFrame(summary_data)
output_csv = paper_output_path / "economic_surplus_summary.csv"
df_summary.to_csv(output_csv, index=False, float_format="%.1f")
logger.info(f"Saved surplus summary to: {output_csv}")

# Table 2: By-area breakdown
area_data = []
for scenario_name in scenario_order:
    if scenario_name in surplus_results:
        by_area = surplus_results[scenario_name]["by_area"]
        for area in ALL_AREAS:
            if area in by_area:
                area_data.append(
                    {
                        "Scenario": scenario_name,
                        "Area": area,
                        "Consumer Surplus (M€)": by_area[area]["consumer_surplus"],
                        "Producer Surplus (M€)": by_area[area]["producer_surplus"],
                        "Societal Surplus (M€)": by_area[area]["societal_surplus"],
                        "Avg Price (€/MWh)": by_area[area]["avg_price"],
                    }
                )

df_area = pd.DataFrame(area_data)
output_csv_area = paper_output_path / "economic_surplus_by_area.csv"
df_area.to_csv(output_csv_area, index=False, float_format="%.2f")
logger.info(f"Saved area-level surplus to: {output_csv_area}")

# ============================================================================
# Create LaTeX tables
# ============================================================================

def _resolve_scenario_key(candidates: list[str]) -> str | None:
    for key in candidates:
        if key in surplus_results:
            return key
    return None


ROW_SPECS = [
    ("N-BA+", ["BA_N", "N-BA+"]),
    ("OWN-BA+", ["BA_OWN", "OWN-BA+"]),
    ("OW-BA+", ["BA_OW", "OW-BA+"]),
    ("N-LLPS+", ["LLPS_N", "N-LLPS+"]),
    ("OWN-LLPS+", ["LLPS_OWN", "OWN-LLPS+"]),
    ("OW-LLPS+", ["LLPS_OW", "OW-LLPS+"]),
]


# LaTeX Table 1: Norway results
latex_table_norway = r"""\begin{table}[htbp]
\centering
\caption{Economic surplus for Norway. Baseline in billion EUR, scenarios show change in billion EUR (percentage change).}
\label{tab:surplus_norway}
\begin{tabular}{lrrr}
\toprule
Scenario & Consumer & Producer & Societal \\
 & Surplus & Surplus & Surplus \\
\midrule
"""

# Add baseline first
if baseline_short in surplus_results:
    baseline_no = surplus_results[baseline_short]["norway_total"]
    cs_base = baseline_no["consumer_surplus"] / 1000
    ps_base = baseline_no["producer_surplus"] / 1000
    ss_base = baseline_no["societal_surplus"] / 1000

    latex_table_norway += f"\\multicolumn{{1}}{{l}}{{\\texttt{{B}}}} & {cs_base:.1f} & {ps_base:.1f} & {ss_base:.1f} \\\\\n"
    latex_table_norway += r"\midrule" + "\n"

# Add scenarios in fixed display order and formatting
for idx, (display_label, candidates) in enumerate(ROW_SPECS):
    scenario_name = _resolve_scenario_key(candidates)
    if scenario_name is None:
        continue

    no_total = surplus_results[scenario_name]["norway_total"]
    baseline_no = surplus_results[baseline_short]["norway_total"]

    cs_delta = (no_total["consumer_surplus"] - baseline_no["consumer_surplus"]) / 1000
    ps_delta = (no_total["producer_surplus"] - baseline_no["producer_surplus"]) / 1000
    ss_delta = (no_total["societal_surplus"] - baseline_no["societal_surplus"]) / 1000

    cs_pct = (
        ((no_total["consumer_surplus"] - baseline_no["consumer_surplus"]) / baseline_no["consumer_surplus"] * 100)
        if baseline_no["consumer_surplus"] != 0
        else 0
    )
    ps_pct = (
        ((no_total["producer_surplus"] - baseline_no["producer_surplus"]) / baseline_no["producer_surplus"] * 100)
        if baseline_no["producer_surplus"] != 0
        else 0
    )
    ss_pct = (
        ((no_total["societal_surplus"] - baseline_no["societal_surplus"]) / baseline_no["societal_surplus"] * 100)
        if baseline_no["societal_surplus"] != 0
        else 0
    )

    cs_str = f"{cs_delta:+.1f} ({cs_pct:+.0f}\\%)"
    ps_str = f"{ps_delta:+.1f} ({ps_pct:+.0f}\\%)"
    ss_str = f"{ss_delta:+.1f} ({ss_pct:+.0f}\\%)"
    latex_table_norway += f"\\texttt{{{display_label}}} & {cs_str} & {ps_str} & {ss_str} \\\\\n"
    if idx == 2:
        latex_table_norway += r"\midrule" + "\n"

latex_table_norway += r"""\bottomrule
\end{tabular}
\end{table}
"""

# LaTeX Table 2: All Nordic results
latex_table_nordic = r"""\begin{table}[htbp]
\centering
\caption{Economic surplus for all Nordic areas. Baseline in billion EUR, scenarios show change in billion EUR (percentage change).}
\label{tab:surplus_nordic}
\begin{tabular}{lrrr}
\toprule
Scenario & Consumer & Producer & Societal \\
 & Surplus & Surplus & Surplus \\
\midrule
"""

# Add baseline first
if baseline_short in surplus_results:
    baseline_all = surplus_results[baseline_short]["all_areas_total"]
    cs_base = baseline_all["consumer_surplus"] / 1000
    ps_base = baseline_all["producer_surplus"] / 1000
    ss_base = baseline_all["societal_surplus"] / 1000

    latex_table_nordic += f"\\multicolumn{{1}}{{l}}{{\\texttt{{B}}}} & {cs_base:.1f} & {ps_base:.1f} & {ss_base:.1f} \\\\\n"
    latex_table_nordic += r"\midrule" + "\n"

# Add scenarios in fixed display order and formatting
for idx, (display_label, candidates) in enumerate(ROW_SPECS):
    scenario_name = _resolve_scenario_key(candidates)
    if scenario_name is None:
        continue

    all_total = surplus_results[scenario_name]["all_areas_total"]
    baseline_all = surplus_results[baseline_short]["all_areas_total"]

    cs_delta = (all_total["consumer_surplus"] - baseline_all["consumer_surplus"]) / 1000
    ps_delta = (all_total["producer_surplus"] - baseline_all["producer_surplus"]) / 1000
    ss_delta = (all_total["societal_surplus"] - baseline_all["societal_surplus"]) / 1000

    cs_pct = (
        ((all_total["consumer_surplus"] - baseline_all["consumer_surplus"]) / baseline_all["consumer_surplus"] * 100)
        if baseline_all["consumer_surplus"] != 0
        else 0
    )
    ps_pct = (
        ((all_total["producer_surplus"] - baseline_all["producer_surplus"]) / baseline_all["producer_surplus"] * 100)
        if baseline_all["producer_surplus"] != 0
        else 0
    )
    ss_pct = (
        ((all_total["societal_surplus"] - baseline_all["societal_surplus"]) / baseline_all["societal_surplus"] * 100)
        if baseline_all["societal_surplus"] != 0
        else 0
    )

    cs_str = f"{cs_delta:+.1f} ({cs_pct:+.0f}\\%)"
    ps_str = f"{ps_delta:+.1f} ({ps_pct:+.0f}\\%)"
    ss_str = f"{ss_delta:+.1f} ({ss_pct:+.0f}\\%)"
    latex_table_nordic += f"\\texttt{{{display_label}}} & {cs_str} & {ps_str} & {ss_str} \\\\\n"
    if idx == 2:
        latex_table_nordic += r"\midrule" + "\n"

latex_table_nordic += r"""\bottomrule
\end{tabular}
\end{table}
"""

# Save LaTeX tables
latex_output_norway = paper_output_path / "surplus_table_norway.tex"
with open(latex_output_norway, "w") as f:
    f.write(latex_table_norway)
logger.info(f"Saved Norway LaTeX table to: {latex_output_norway}")

latex_output_nordic = paper_output_path / "surplus_table_nordic.tex"
with open(latex_output_nordic, "w") as f:
    f.write(latex_table_nordic)
logger.info(f"Saved Nordic LaTeX table to: {latex_output_nordic}")

# ============================================================================
# Create technology-level producer surplus comparison
# ============================================================================

# Create DataFrame for technology surplus comparison
tech_comparison_data = []
for scenario_name in scenario_order:
    if scenario_name in technology_surplus_results:
        tech_data = technology_surplus_results[scenario_name]
        row = {
            "Scenario": _display_scenario_name(scenario_name),
            "Hydro (M€)": tech_data["hydro"],
            "Solar (M€)": tech_data["solar"],
            "Wind Onshore (M€)": tech_data["wind_onshore"],
            "Wind Offshore (M€)": tech_data["wind_offshore"],
            "Nuclear (M€)": tech_data["nuclear"],
            "Biomass (M€)": tech_data["biomass"],
            "Fossil Gas (M€)": tech_data["fossil_gas"],
            "Fossil Other (M€)": tech_data["fossil_other"],
            "Total (M€)": sum(tech_data.values()),
        }
        tech_comparison_data.append(row)

df_tech = pd.DataFrame(tech_comparison_data)
output_csv_tech = paper_output_path / "producer_surplus_by_technology.csv"
df_tech.to_csv(output_csv_tech, index=False, float_format="%.1f")
logger.info(f"Saved technology-level surplus to: {output_csv_tech}")

# Create LaTeX table for technology comparison
latex_tech_table = r"""\begin{table}[htbp]
\centering
\caption{Producer Surplus by Technology for Norway (M€)}
\label{tab:producer_surplus_technology}
\begin{tabular}{lrrrrrrrrr}
\toprule
Scenario & Hydro & Solar & Wind & Wind & Nuclear & Biomass & Fossil & Fossil & Total \\
         &       &       & Onshore & Offshore &  &  & Gas & Other &  \\
\midrule
"""

# Add baseline first
if baseline_short in technology_surplus_results:
    base_tech = technology_surplus_results[baseline_short]
    latex_tech_table += (
        f"\\texttt{{B}} & {base_tech['hydro']:.0f} & {base_tech['solar']:.0f} & "
        f"{base_tech['wind_onshore']:.0f} & {base_tech['wind_offshore']:.0f} & "
        f"{base_tech['nuclear']:.0f} & {base_tech['biomass']:.0f} & "
        f"{base_tech['fossil_gas']:.0f} & {base_tech['fossil_other']:.0f} & "
        f"{sum(base_tech.values()):.0f} \\\\\n"
    )
    latex_tech_table += r"\midrule" + "\n"

# Add scenarios in the same display order as surplus tables
for idx, (display_label, candidates) in enumerate(ROW_SPECS):
    scenario_name = _resolve_scenario_key(candidates)
    if scenario_name is None or scenario_name not in technology_surplus_results:
        continue

    tech_data = technology_surplus_results[scenario_name]
    baseline_tech = technology_surplus_results[baseline_short]

    hydro_delta = tech_data["hydro"] - baseline_tech["hydro"]
    solar_delta = tech_data["solar"] - baseline_tech["solar"]
    wind_on_delta = tech_data["wind_onshore"] - baseline_tech["wind_onshore"]
    wind_off_delta = tech_data["wind_offshore"] - baseline_tech["wind_offshore"]
    nuclear_delta = tech_data["nuclear"] - baseline_tech["nuclear"]
    biomass_delta = tech_data["biomass"] - baseline_tech["biomass"]
    fossil_gas_delta = tech_data["fossil_gas"] - baseline_tech["fossil_gas"]
    fossil_other_delta = tech_data["fossil_other"] - baseline_tech["fossil_other"]
    total_delta = sum(tech_data.values()) - sum(baseline_tech.values())

    latex_tech_table += (
        f"\\texttt{{{display_label}}} & "
        f"{hydro_delta:+.0f} & {solar_delta:+.0f} & "
        f"{wind_on_delta:+.0f} & {wind_off_delta:+.0f} & "
        f"{nuclear_delta:+.0f} & {biomass_delta:+.0f} & "
        f"{fossil_gas_delta:+.0f} & {fossil_other_delta:+.0f} & "
        f"{total_delta:+.0f} \\\\\n"
    )
    if idx == 2:
        latex_tech_table += r"\midrule" + "\n"

latex_tech_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

# Save technology LaTeX table
latex_output_tech = paper_output_path / "surplus_table_technology.tex"
with open(latex_output_tech, "w") as f:
    f.write(latex_tech_table)
logger.info(f"Saved technology LaTeX table to: {latex_output_tech}")

# Print summary
print("\n" + "=" * 100)
print("ECONOMIC SURPLUS SUMMARY")
print("=" * 100)
print(df_summary.to_string(index=False))
print("=" * 100)

print("\n" + "=" * 100)
print("LATEX TABLE - NORWAY")
print("=" * 100)
print(latex_table_norway)
print("=" * 100)

print("\n" + "=" * 100)
print("LATEX TABLE - ALL NORDIC")
print("=" * 100)
print(latex_table_nordic)
print("=" * 100)

print("\n" + "=" * 100)
print("PRODUCER SURPLUS BY TECHNOLOGY (NORWAY)")
print("=" * 100)
print(df_tech.to_string(index=False))
print("=" * 100)

print("\n" + "=" * 100)
print("LATEX TABLE - TECHNOLOGY COMPARISON")
print("=" * 100)
print(latex_tech_table)
print("=" * 100)

plt.show()
