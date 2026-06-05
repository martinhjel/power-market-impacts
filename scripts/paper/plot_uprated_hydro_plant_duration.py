"""
Plot duration curves for hydropower reservoirs connected to uprated plants.

This script reads processed reservoir production from
ltm_processed/<model>/<scenario>/processed_data.parquet and compares the
production duration curves for the reservoirs listed in data/uprate_hydro.py.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.append(str(DATA_DIR))

from scripts.common import load_scenarios, logger
from uprate_hydro import uprate_values as UPRATED_PLANTS  # type: ignore

MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"
NO_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

SHORT_TO_SCENARIO = {
    "B": "BASELINE_00TWh_FalseHYD_FalseFF_BALOAD_0.00TWH_NoneNUKE_NoneOFF",
    "B+": "BASELINE_UPRATE_TrueHYD_FalseFF_NONELOAD_0.00TWH_NoneNUKE_NoneOFF",
    "N-LLPS": "LLPS_N_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "N-LLPS+": "LLPS_N_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-LLPS": "LLPS_OWN_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OWN-LLPS+": "LLPS_OWN_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-LLPS": "LLPS_OW_FalseHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "OW-LLPS+": "LLPS_OW_UPRATE_TrueHYD_FalseFF_LLPSLOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "N-BA": "BA_N_FalseHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "N-BA+": "BA_N_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_900p8407111111112NO2-2005p0970666666665NO1NUKE_NoneOFF",
    "OWN-BA": "BA_OWN_FalseHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OWN-BA+": "BA_OWN_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_1781p3768888888892NO1NUKE_1400NO2-500NO2OFF",
    "OW-BA": "BA_OW_FalseHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
    "OW-BA+": "BA_OW_UPRATE_TrueHYD_FalseFF_BALOAD_22.91TWH_NoneNUKE_3000NO2-500NO2-1500NO5OFF",
}

SCENARIO_SETS = {
    "baseline": ["B", "B+"],
    "ba": ["B", "B+", "N-BA", "N-BA+", "OWN-BA", "OWN-BA+", "OW-BA", "OW-BA+"],
    "llps": ["B", "B+", "N-LLPS", "N-LLPS+", "OWN-LLPS", "OWN-LLPS+", "OW-LLPS", "OW-LLPS+"],
    "all": [
        "B",
        "B+",
        "N-BA",
        "N-BA+",
        "OWN-BA",
        "OWN-BA+",
        "OW-BA",
        "OW-BA+",
        "N-LLPS",
        "N-LLPS+",
        "OWN-LLPS",
        "OWN-LLPS+",
        "OW-LLPS",
        "OW-LLPS+",
    ],
}

SCENARIO_COLORS = {
    "B": "#7f7f7f",
    "N-LLPS": "#1f77b4",
    "OWN-LLPS": "#2ca02c",
    "OW-LLPS": "#17becf",
    "N-BA": "#ff7f0e",
    "OWN-BA": "#d62728",
    "OW-BA": "#ff9896",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reservoir-level duration curves for uprated hydropower plants."
    )
    parser.add_argument("--model-folder", default=MODEL_FOLDER)
    parser.add_argument(
        "--scenario-set",
        choices=sorted(SCENARIO_SETS),
        default="ba",
        help="Predefined scenario group to compare.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SHORT_TO_SCENARIO),
        help="Explicit short scenario labels. Overrides --scenario-set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to visualizations/<model-folder>/paper.",
    )
    parser.add_argument(
        "--max-plants",
        type=int,
        help="Limit the number of plant panels, useful for quick checks.",
    )
    parser.add_argument(
        "--plants",
        nargs="+",
        choices=sorted(UPRATED_PLANTS),
        help="Only plot selected uprated plants, for example: --plants kvanndal.",
    )
    return parser.parse_args()


def scenario_order(args: argparse.Namespace) -> list[str]:
    return args.scenarios if args.scenarios else SCENARIO_SETS[args.scenario_set]


def scenario_paths(model_folder: str, short_names: list[str]) -> dict[str, Path]:
    paths = {}
    for short_name in short_names:
        scenario_name = SHORT_TO_SCENARIO[short_name]
        path = PROJECT_ROOT / "ltm_output" / model_folder / scenario_name
        if not path.exists():
            logger.warning("Skipping missing scenario %s at %s", short_name, path)
            continue
        paths[scenario_name] = path
    return paths


def reservoir_targets() -> dict[str, dict[str, str]]:
    targets = {}
    for plant_name, info in UPRATED_PLANTS.items():
        reservoirs = info.get("reservoirs", [])
        if not reservoirs:
            continue
        reservoir_name = reservoirs[0]
        targets[plant_name] = {
            "reservoir_name": reservoir_name,
            "reservoir_ltm_names": [
                reservoir_name.lower(),
                f"res_{reservoir_name.lower()}",
                f"reservoir_{reservoir_name.lower()}",
            ],
            "area": info.get("elspot_area", ""),
        }
    return targets


def normalized_reservoir_name(name: str) -> str:
    name = name.lower()
    for prefix in ("reservoir_", "res_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return re.sub(r"_[0-9a-f]{4}$", "", name)


def find_reservoir(reservoirs: dict[str, object], ltm_names: list[str]) -> object | None:
    for name in ltm_names:
        reservoir = reservoirs.get(str(name).lower())
        if reservoir is not None:
            return reservoir

    target = normalized_reservoir_name(str(ltm_names[0]))
    if len(target) < 5:
        return None

    candidates = [
        reservoir
        for name, reservoir in reservoirs.items()
        if normalized_reservoir_name(name).startswith(target[:5])
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def collect_reservoir_production(
    scenario,
    selected_plants: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    targets = reservoir_targets()
    if selected_plants is not None:
        targets = {plant: target for plant, target in targets.items() if plant in selected_plants}
    found = {}

    for plant_name, target in targets.items():
        try:
            df_prod = scenario.get_reservoir_production(plant_name)
        except Exception:
            continue
        found[plant_name] = {
            "area": target["area"],
            "reservoir_name": target["reservoir_name"],
            "generation": df_prod,
        }

    missing = sorted(set(targets) - set(found))
    if missing:
        logger.warning(
            "Missing %d processed uprated reservoir production series in %s: %s",
            len(missing),
            scenario.name,
            ", ".join(missing),
        )
    return found


def duration_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = df.to_numpy(dtype=float).flatten()
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.array([]), np.array([])
    y = np.sort(values)[::-1]
    x = np.linspace(0, 1, len(y))
    return x, y


def scenario_family(short_name: str) -> str:
    return short_name[:-1] if short_name.endswith("+") else short_name


def plot_duration_curves(
    plant_results: dict[str, dict[str, dict[str, object]]],
    short_names: list[str],
    output_file: Path,
) -> None:
    plant_names = sorted(plant_results)
    ncols = 3
    nrows = math.ceil(len(plant_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, plant_name in zip(axes_flat, plant_names):
        plant_data = plant_results[plant_name]
        for short_name in short_names:
            scenario_name = SHORT_TO_SCENARIO[short_name]
            if scenario_name not in plant_data:
                continue

            generation = plant_data[scenario_name]["generation"]
            x, y = duration_curve(generation)
            if len(y) == 0:
                continue

            family = scenario_family(short_name)
            ax.plot(
                x,
                y,
                label=short_name,
                color=SCENARIO_COLORS.get(family, "gray"),
                linestyle="solid" if short_name.endswith("+") else ":",
                linewidth=2.2 if short_name.endswith("+") else 1.8,
                alpha=0.88,
            )

        area = next(iter(plant_data.values())).get("area", "")
        reservoir_name = next(iter(plant_data.values())).get("reservoir_name", "")
        ax.set_title(f"{plant_name} ({area}, {reservoir_name})", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of hours")
        ax.set_ylabel("Generation (MW)")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(plant_names) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7), framealpha=0.95)
    fig.suptitle("Duration Curves for Uprated Hydropower Reservoirs", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_stats_rows(plant_results: dict[str, dict[str, dict[str, object]]]) -> list[dict[str, object]]:
    rows = []
    scenario_to_short = {v: k for k, v in SHORT_TO_SCENARIO.items()}
    for plant_name, plant_data in sorted(plant_results.items()):
        for scenario_name, entry in plant_data.items():
            df = entry["generation"]
            values = df.to_numpy(dtype=float).flatten()
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue
            rows.append(
                {
                    "plant": plant_name,
                    "reservoir": entry["reservoir_name"],
                    "area": entry["area"],
                    "scenario": scenario_to_short.get(scenario_name, scenario_name),
                    "mean_mw": float(np.mean(values)),
                    "max_mw": float(np.max(values)),
                    "p95_mw": float(np.percentile(values, 95)),
                    "p99_mw": float(np.percentile(values, 99)),
                    "zero_fraction": float(np.mean(values <= 1e-9)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    short_names = scenario_order(args)
    output_dir = args.output_dir or PROJECT_ROOT / OUTPUT_DIR / args.model_folder / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = scenario_paths(args.model_folder, short_names)
    if not paths:
        raise SystemExit("No scenario folders found")

    scenarios = load_scenarios(paths)
    if not scenarios:
        raise SystemExit("No scenarios loaded")

    plant_results: dict[str, dict[str, dict[str, object]]] = {}
    logger.info("Collecting reservoir production for %d scenarios", len(scenarios))
    for scenario_name, scenario in scenarios.items():
        short_name = {v: k for k, v in SHORT_TO_SCENARIO.items()}.get(scenario_name, scenario_name)
        logger.info("Processing %s", short_name)
        scenario_results = collect_reservoir_production(
            scenario,
            selected_plants=set(args.plants) if args.plants else None,
        )
        for plant_name, result in scenario_results.items():
            plant_results.setdefault(plant_name, {})[scenario_name] = result

    if args.plants:
        requested = set(args.plants)
        plant_results = {plant: data for plant, data in plant_results.items() if plant in requested}

    if args.max_plants is not None:
        limited = sorted(plant_results)[: args.max_plants]
        plant_results = {plant: plant_results[plant] for plant in limited}

    if not plant_results:
        raise SystemExit("No uprated reservoir production found")

    set_name = "custom" if args.scenarios else args.scenario_set
    if args.plants:
        set_name = f"{set_name}_{'_'.join(args.plants)}"
    pdf_path = output_dir / f"uprated_hydro_plant_duration_{set_name}.pdf"
    csv_path = output_dir / f"uprated_hydro_plant_duration_stats_{set_name}.csv"

    plot_duration_curves(plant_results, short_names, pdf_path)
    pd.DataFrame(build_stats_rows(plant_results)).to_csv(csv_path, index=False)

    logger.info("Saved duration curves to %s", pdf_path)
    logger.info("Saved duration statistics to %s", csv_path)


if __name__ == "__main__":
    main()
