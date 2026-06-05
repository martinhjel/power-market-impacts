"""
Generate a LaTeX table with an overview of all scenarios (case studies).
Uses scenario definitions from scenario_runner.py to get accurate nuclear and offshore wind locations.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Add parent directory to path to import scenario_runner
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scenario_runner import SCENARIOS as SCENARIO_CONFIGS

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Setup paths
base_path = Path.cwd()
config_file = base_path / "config/visualization_config.yaml"
output_path = base_path / OUTPUT_DIR / MODEL_FOLDER
paper_output_path = output_path / "paper"
paper_output_path.mkdir(parents=True, exist_ok=True)

# Load configuration
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Get all unique scenarios across all groups
all_scenarios = set()
for group_scenarios in config["custom_groups"].values():
    all_scenarios.update(group_scenarios)

all_scenarios = sorted(all_scenarios)

print(f"Found {len(all_scenarios)} unique scenarios")


def format_nuclear_info(sc):
    """Format nuclear additions into readable string."""
    if not sc.nuclear_additions:
        return "None"

    total_cap = sum(n["capacity"] for n in sc.nuclear_additions)
    areas = [n["area"] for n in sc.nuclear_additions]

    if len(areas) == 1:
        return f"{total_cap:.0f} MW in {areas[0]}"
    elif len(areas) == 2:
        caps = [f"{n['capacity']:.0f} MW" for n in sc.nuclear_additions]
        return f"{areas[0]} ({caps[0]}), {areas[1]} ({caps[1]})"
    else:
        # Multiple areas
        area_str = ", ".join(areas)
        return f"{total_cap:.0f} MW ({area_str})"


def format_offshore_info(sc):
    """Format offshore wind additions into readable string."""
    if not sc.offshore_wind_additions:
        return "None"

    total_cap = sum(o["capacity"] for o in sc.offshore_wind_additions)
    areas = [o["connected_to"] for o in sc.offshore_wind_additions]

    if len(areas) == 1:
        return f"{total_cap:.0f} MW in {areas[0]}"
    elif len(areas) == 2:
        caps = [f"{o['capacity']:.0f} MW" for o in sc.offshore_wind_additions]
        return f"{areas[0]} ({caps[0]}), {areas[1]} ({caps[1]})"
    else:
        # Multiple areas
        area_str = ", ".join(set(areas))
        return f"{total_cap:.0f} MW ({area_str})"


def parse_scenario_name(scenario_name):
    """
    Parse scenario name to extract key parameters.

    Example formats:
    - BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF
    - SMR300BA_20TWh_FalseHYD_FalseFF_BALOAD_30.00TWH_300NO1-300NO2-300NO3-300NO4-300NO5NUKE_NoneOFF
    - BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF
    """

    parts = {}

    # Extract case type (first part before first underscore or TWh)
    if scenario_name.startswith("BASELINE"):
        parts["case"] = "BASELINE"
    elif scenario_name.startswith("SMR"):
        match = re.match(r"SMR(\d+)(BA|LLPS)", scenario_name)
        if match:
            parts["case"] = f"SMR-{match.group(1)}"
            parts["load_profile"] = match.group(2)
    elif scenario_name.startswith("LMR"):
        match = re.match(r"LMR(\d+)(BA|LLPS)", scenario_name)
        if match:
            parts["case"] = f"LMR-{match.group(1)}"
            parts["load_profile"] = match.group(2)
    elif scenario_name.startswith("BA_"):
        parts["case"] = "BA"
        parts["load_profile"] = "BA"
    elif scenario_name.startswith("LLPS_"):
        parts["case"] = "LLPS"
        parts["load_profile"] = "LLPS"

    # Extract if it has UPRATE
    parts["hydro_uprate"] = "TrueHYD" in scenario_name or "UPRATE" in scenario_name

    # Extract load increment
    load_match = re.search(r"_(\d+)TWh_", scenario_name)
    if load_match:
        parts["load_increment"] = int(load_match.group(1))

    # Extract total load
    total_load_match = re.search(r"LOAD_(\d+\.?\d*)TWH", scenario_name)
    if total_load_match:
        parts["total_load"] = float(total_load_match.group(1))

    # Extract nuclear configuration
    if "_NoneNUKE_" in scenario_name:
        parts["nuclear"] = "None"
    else:
        # Extract nuclear details
        nuke_match = re.search(r"NUKE_(.+?)OFF", scenario_name)
        if nuke_match:
            nuke_str = nuke_match.group(1).replace("_None", "")
            # Count number of locations
            locations = nuke_str.split("-")
            if locations and locations[0]:
                parts["nuclear"] = f"{len(locations)} locations"

                # Try to extract total capacity
                capacity_sum = 0
                for loc in locations:
                    cap_match = re.match(r"(\d+)", loc)
                    if cap_match:
                        capacity_sum += int(cap_match.group(1))
                if capacity_sum > 0:
                    parts["nuclear_capacity"] = capacity_sum

    # Extract offshore wind configuration
    if "_NoneOFF" in scenario_name:
        parts["offshore"] = "None"
    else:
        off_match = re.search(r"NUKE_(.+)OFF", scenario_name)
        if off_match:
            off_str = off_match.group(1)
            if "OFF" in off_str:
                # Extract offshore details
                locations = [x for x in off_str.split("-") if "NO" in x or "SE" in x or "DK" in x]
                if locations:
                    parts["offshore"] = f"{len(locations)} locations"

    # Detect technology type from case name
    if "N_" in scenario_name or scenario_name.startswith("LMR") or scenario_name.startswith("SMR"):
        parts["technology"] = "Nuclear"
    elif "OWN_" in scenario_name:
        parts["technology"] = "Nuclear + Offshore"
    elif "OW_" in scenario_name:
        parts["technology"] = "Offshore Wind"
    elif "BASELINE" in scenario_name:
        parts["technology"] = "Baseline"

    return parts


# Create a lookup dictionary from scenario_runner.py using just the name field
scenario_lookup = {}
for sc in SCENARIO_CONFIGS:
    # Use only the name field from the config
    scenario_lookup[sc.name] = sc

print(f"Loaded {len(scenario_lookup)} scenario definitions from scenario_runner.py")

# Create a mapping from full scenario names to short names
name_mapping = {}
for full_name in all_scenarios:
    # Extract the short name (first part before first underscore or the whole name)
    # This matches patterns like "BASELINE_00TWh" -> "BASELINE_00TWh"
    # or "BA_N_FalseHYD..." -> "BA_N"
    parts = full_name.split("_")

    # Try different combinations to find a match
    for i in range(len(parts), 0, -1):
        short_name = "_".join(parts[:i])
        if short_name in scenario_lookup:
            name_mapping[full_name] = short_name
            break

    # If no match found, check if any scenario name is a prefix
    if full_name not in name_mapping:
        for sc_name in scenario_lookup.keys():
            if full_name.startswith(sc_name + "_"):
                name_mapping[full_name] = sc_name
                break

print(f"Mapped {len(name_mapping)} scenarios to their definitions")


# Parse all scenarios
scenario_data = []
for scenario_name in all_scenarios:
    parsed = parse_scenario_name(scenario_name)

    # Try to find matching config from scenario_runner.py
    short_name = scenario_name  # Default to full name
    if scenario_name in name_mapping:
        short_name = name_mapping[scenario_name]
        sc = scenario_lookup[short_name]
        parsed["nuclear_detail"] = format_nuclear_info(sc)
        parsed["offshore_detail"] = format_offshore_info(sc)
        parsed["load_mode"] = sc.load_mode.value if hasattr(sc.load_mode, "value") else str(sc.load_mode)
        parsed["total_load"] = sc.additional_load_twh
        parsed["hydro_uprate"] = sc.uprate_hydro

    # Use short name for display
    parsed["name"] = short_name
    parsed["full_name"] = scenario_name

    scenario_data.append(parsed)


# Group scenarios by case study
case_groups = defaultdict(list)
for data in scenario_data:
    case = data.get("case", "Other")
    case_groups[case].append(data)


# Create output directory
output_file = paper_output_path / "scenario_overview_table.tex"


# Generate LaTeX table
latex_lines = []
latex_lines.append("\\begin{table}[htbp]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Overview of Scenario Case Studies}")
latex_lines.append("\\label{tab:scenario_overview}")
latex_lines.append("\\small")
latex_lines.append("\\begin{tabular}{lllccll}")
latex_lines.append("\\hline")
latex_lines.append(
    "\\textbf{Case} & \\textbf{Technology} & \\textbf{Load Profile} & \\textbf{Load (TWh)} & \\textbf{Hydro Uprate} & \\textbf{Nuclear} & \\textbf{Offshore Wind} \\\\"
)
latex_lines.append("\\hline")

# Sort case groups for logical presentation
case_order = ["BASELINE", "BA", "LLPS"]
case_order.extend([k for k in sorted(case_groups.keys()) if k not in case_order])

for case in case_order:
    if case not in case_groups:
        continue

    scenarios = case_groups[case]

    # Sort scenarios within each case by load, then by uprate
    scenarios.sort(key=lambda x: (x.get("load_increment", 0), x.get("total_load", 0), not x.get("hydro_uprate", False)))

    for i, data in enumerate(scenarios):
        # Case name (only for first row)
        if i == 0:
            case_name = case
        else:
            case_name = ""

        # Technology
        tech = data.get("technology", "N/A")

        # Load profile
        load_profile = data.get(
            "load_profile",
            "BA" if "BA" in data.get("name", "") else "LLPS" if "LLPS" in data.get("name", "") else "N/A",
        )

        # Total load
        total_load = data.get("total_load", data.get("load_increment", 0))
        load_str = f"{total_load:.1f}" if total_load > 0 else "N/A"

        # Hydro uprate
        uprate = "Yes" if data.get("hydro_uprate", False) else "No"

        # Nuclear capacity with location details
        nuke_str = data.get("nuclear_detail", data.get("nuclear", "None"))

        # Offshore wind with location details
        offshore = data.get("offshore_detail", data.get("offshore", "None"))

        latex_lines.append(
            f"{case_name} & {tech} & {load_profile} & {load_str} & {uprate} & {nuke_str} & {offshore} \\\\"
        )

    # Add separator between case groups
    if case != case_order[-1]:
        latex_lines.append("\\hline")

latex_lines.append("\\hline")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")

# Write to file
with open(output_file, "w") as f:
    f.write("\n".join(latex_lines))

print(f"\nLaTeX table saved to: {output_file}")
print(f"Total scenarios: {len(scenario_data)}")
print("\nCase breakdown:")
for case in case_order:
    if case in case_groups:
        print(f"  {case}: {len(case_groups[case])} scenarios")

# Also generate a detailed table with full scenario names
output_file_detailed = paper_output_path / "scenario_overview_detailed.tex"

latex_lines_detailed = []
latex_lines_detailed.append("\\begin{landscape}")
latex_lines_detailed.append("\\begin{table}[htbp]")
latex_lines_detailed.append("\\centering")
latex_lines_detailed.append("\\caption{Detailed Scenario Overview with Full Names}")
latex_lines_detailed.append("\\label{tab:scenario_overview_detailed}")
latex_lines_detailed.append("\\tiny")
latex_lines_detailed.append("\\begin{tabular}{lp{8cm}lcccc}")
latex_lines_detailed.append("\\hline")
latex_lines_detailed.append(
    "\\textbf{ID} & \\textbf{Scenario Name} & \\textbf{Technology} & \\textbf{Load (TWh)} & \\textbf{Uprate} & \\textbf{Nuclear} & \\textbf{Offshore} \\\\"
)
latex_lines_detailed.append("\\hline")

for idx, data in enumerate(scenario_data, 1):
    name = data["name"]
    # Make name more readable for LaTeX (escape underscores)
    name_latex = name.replace("_", "\\_")

    tech = data.get("technology", "N/A")
    total_load = data.get("total_load", data.get("load_increment", 0))
    load_str = f"{total_load:.1f}" if total_load > 0 else "N/A"
    uprate = "Y" if data.get("hydro_uprate", False) else "N"
    nuke = "Y" if data.get("nuclear", "None") != "None" else "N"
    offshore = "Y" if data.get("offshore", "None") != "None" else "N"

    latex_lines_detailed.append(
        f"{idx} & \\texttt{{{name_latex}}} & {tech} & {load_str} & {uprate} & {nuke} & {offshore} \\\\"
    )

latex_lines_detailed.append("\\hline")
latex_lines_detailed.append("\\end{tabular}")
latex_lines_detailed.append("\\end{table}")
latex_lines_detailed.append("\\end{landscape}")

with open(output_file_detailed, "w") as f:
    f.write("\n".join(latex_lines_detailed))

print(f"\nDetailed table saved to: {output_file_detailed}")
