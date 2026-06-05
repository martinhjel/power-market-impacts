"""
Generate metadata file with scenario information.
"""

import json
from pathlib import Path

# Snakemake inputs/outputs
output_file = Path(snakemake.output[0])
model_folder = snakemake.params.model_folder
scenarios = snakemake.params.scenarios

# Create output directory
output_file.parent.mkdir(parents=True, exist_ok=True)

# Build scenario paths
ltm_output = Path("ltm_output")
base_path = ltm_output / model_folder

scenario_paths = {}
for scenario in scenarios:
    scenario_path = base_path / scenario
    if scenario_path.exists():
        scenario_paths[scenario] = str(scenario_path)

# Save metadata
metadata = {"model_folder": model_folder, "base_path": str(base_path), "scenarios": scenario_paths}

with open(output_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Generated metadata for {len(scenario_paths)} scenarios")

print(f"Generated metadata for {len(scenario_paths)} scenarios")
