"""
Script to read reservoir capacities from dataset.json.

Extracts reservoir_capacity_mm3 and global_energy_equivalent for all reservoirs
organized by busbar/area.
"""

import json
from collections import defaultdict
from pathlib import Path


def read_reservoir_capacities(dataset_path: Path):
    """
    Read reservoir capacities from dataset.json.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset.json file

    Returns
    -------
    dict
        Dictionary mapping busbar names to lists of reservoir data
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    reservoir_data = {}
    for item in dataset["reservoirs"]["items"]:
        name = item["name"]
        name = name.replace(".","")
        reservoir_data[name] = {
            "reservoir_capacity_mm3": item["metadata"]["reservoir_capacity_mm3"],
            "global_energy_equivalent": item["metadata"]["global_energy_equivalent"],
        }
        
    return reservoir_data


if __name__ == "__main__":
    # Path to dataset.json
    model_folder = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO"

    dataset_path = Path(f"ltm_output/{model_folder}/dataset.json")

    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        print("\nPlease update the model_folder and scenario_folder variables in the script.")
        exit(1)

    print(f"Reading dataset from: {dataset_path}")

    reservoir_data = read_reservoir_capacities(dataset_path)

    # Optional: Save to JSON for programmatic access
    output_file = Path("data/reservoir_capacities.json")
    with open(output_file, "w") as f:
        json.dump(reservoir_data, f, indent=2)
    print(f"Reservoir data saved to: {output_file}")
