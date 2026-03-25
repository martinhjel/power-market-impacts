"""
Plot installed generation capacity stacked by technology for each area
and overlay average load per area from consumer_BM_v100.csv.

Also plot total reservoir trajectories for Norway and Sweden across scenarios.

Saves outputs to `images/area_capacity_vs_load.png` and related PDFs.

Usage:
    python scripts/plot_area_capacity_vs_load.py

Adjust paths at the top of the file if your data is elsewhere.
"""

# For scenario loading
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Paths (adjust if needed)
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "NordicNuclearAnalysis" / "CASE_2025" / "scenario_BM" / "data" / "system" / "combined"
GEN_FILE = DATA_DIR / "generator_BM_v100.csv"
CONS_FILE = DATA_DIR / "consumer_BM_v100.csv"
OUT_DIR = BASE / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not GEN_FILE.exists():
    raise FileNotFoundError(f"Generator file not found: {GEN_FILE}")
if not CONS_FILE.exists():
    raise FileNotFoundError(f"Consumer file not found: {CONS_FILE}")

# Read generator data
gen = pd.read_csv(GEN_FILE, low_memory=False)
# Ensure expected columns
if "node" not in gen.columns or "pmax" not in gen.columns or "type" not in gen.columns:
    raise ValueError(f"Generator CSV missing required columns. Found: {list(gen.columns)}")

# Some rows might have pmax as strings; coerce to numeric
gen["pmax"] = pd.to_numeric(gen["pmax"], errors="coerce").fillna(0.0)

# Aggregate installed capacity by node and technology type
cap = gen.groupby(["node", "type"])["pmax"].sum().unstack(fill_value=0)

# Read consumer/load data
cons = pd.read_csv(CONS_FILE, low_memory=False)
if "node" not in cons.columns or "demand_avg" not in cons.columns:
    raise ValueError(f"Consumer CSV missing required columns. Found: {list(cons.columns)}")
cons["demand_avg"] = pd.to_numeric(cons["demand_avg"], errors="coerce").fillna(0.0)
load = cons.set_index("node")["demand_avg"]

# Determine the set of areas to plot: union of nodes found in consumer file and generator file
areas = sorted(set(load.index).union(set(cap.index)))

# Remove unwanted areas (DE, GB, PL, NL)
exclude_areas = {"DE", "GB", "PL", "NL"}
areas = [a for a in areas if a not in exclude_areas]

# Reindex cap and load to same ordering
cap = cap.reindex(areas, fill_value=0)
load = load.reindex(areas, fill_value=0)

# Order technologies by total capacity (descending) so biggest stacks are at bottom
tech_order = cap.sum(axis=0).sort_values(ascending=False).index.tolist()
cap = cap[tech_order]

# Plot stacked bar
x = np.arange(len(areas))
fig, ax = plt.subplots(figsize=(12, 6))

bottom = np.zeros(len(areas))
colors = plt.get_cmap("tab20").colors
for i, tech in enumerate(tech_order):
    vals = cap[tech].values
    ax.bar(x, vals, bottom=bottom, label=tech, color=colors[i % len(colors)])
    bottom += vals

# Overlay load as black markers/line
ax.plot(x, load.values, marker="o", color="k", linestyle="--", linewidth=2, label="Load (avg) ")

ax.set_xticks(x)
ax.set_xticklabels(areas, rotation=45, ha="right")
ax.set_ylabel("Power (MW)")
ax.set_title("Installed capacity by area (stacked) and average load per area")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
fig.tight_layout()

# Save
png_out = OUT_DIR / "area_capacity_vs_load.png"
pdf_out = OUT_DIR / "area_capacity_vs_load.pdf"
fig.savefig(png_out, dpi=150)
fig.savefig(pdf_out)
print(f"Saved: {png_out}\nSaved: {pdf_out}")


def plot_reservoir_trajectories(scenario_results: dict, output_dir: Path):
    """Plot total reservoir trajectories for Norway and Sweden across scenarios."""
    try:
        # Norwegian busbars
        no_busbars = ["NO1", "NO2", "NO3", "NO4", "NO5"]
        # Swedish busbars
        se_busbars = ["SE1", "SE2", "SE3", "SE4"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot Norway total
        ax = axes[0]
        for scenario_name, scenario in scenario_results.items():
            try:
                total_res = None
                for bb in no_busbars:
                    try:
                        df_res = scenario.get_reservoir_for_busbar(bb)
                        res_mean = df_res.mean(axis=1).values
                        if total_res is None:
                            total_res = res_mean.copy()
                        else:
                            total_res += res_mean
                    except Exception:
                        pass

                if total_res is not None:
                    ax.plot(total_res, label=scenario_name, linewidth=1.5, alpha=0.8)
            except Exception as e:
                print(f"Warning: Could not plot Norway reservoir for {scenario_name}: {e}")

        ax.set_xlabel("Time Period")
        ax.set_ylabel("Total Reservoir Level (MWh)")
        ax.set_title("Norway Total Reservoir Trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        # Plot Sweden total
        ax = axes[1]
        for scenario_name, scenario in scenario_results.items():
            try:
                total_res = None
                for bb in se_busbars:
                    try:
                        df_res = scenario.get_reservoir_for_busbar(bb)
                        res_mean = df_res.mean(axis=1).values
                        if total_res is None:
                            total_res = res_mean.copy()
                        else:
                            total_res += res_mean
                    except Exception:
                        pass

                if total_res is not None:
                    ax.plot(total_res, label=scenario_name, linewidth=1.5, alpha=0.8)
            except Exception as e:
                print(f"Warning: Could not plot Sweden reservoir for {scenario_name}: {e}")

        ax.set_xlabel("Time Period")
        ax.set_ylabel("Total Reservoir Level (MWh)")
        ax.set_title("Sweden Total Reservoir Trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout()

        # Save
        png_out = output_dir / "reservoir_trajectories_norway_sweden.png"
        pdf_out = output_dir / "reservoir_trajectories_norway_sweden.pdf"
        fig.savefig(png_out, dpi=150)
        fig.savefig(pdf_out)
        print(f"Saved: {png_out}\nSaved: {pdf_out}")

    except Exception as e:
        print(f"Could not generate reservoir trajectories: {e}")


if __name__ == "__main__":
    plt.show()

    # Optional: also plot reservoir trajectories if scenario results are available
    # Uncomment and adjust MODEL_FOLDER if you want to include this
    # try:
    #     print("\nAttempting to load scenario results for reservoir trajectories...")
    #     base_path = Path.cwd()
    #     model_folder = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_detFi_IncNOLoad"
    #     scenario_paths = find_scenario_results(base_path, model_folder=model_folder)
    #     if scenario_paths:
    #         scenario_results = {
    #             name: ScenarioResults(path) for name, path in scenario_paths.items()
    #         }
    #         plot_reservoir_trajectories(scenario_results, OUT_DIR)
    # except Exception as e:
    #     print(f"Skipping reservoir plot: {e}")
