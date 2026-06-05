"""
Analyze new nuclear dispatch in Norwegian areas.

This script:
1. Finds new nuclear capacity (identified by marginal cost = 9 €/MWh) in Norwegian areas
2. Fetches hourly prices for each area
3. Computes dispatch: when price > 9, nuclear produces at full capacity; when price <= 9, no production
4. Calculates capacity factor
5. Calculates value factor (average price when dispatched / marginal price)

Usage:
    python scripts/plot_nuclear_dispatch_analysis.py [MODEL_FOLDER]

If MODEL_FOLDER is not provided, uses PowerGamaMSc_2025_BM_1H_serial_TrueEXO_detFi_IncNOLoad
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# For scenario loading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualize_scenarios import ScenarioResults, find_scenario_results

NUCLEAR_MARGINAL_COST = 9.0  # €/MWh
NO_BUSBARS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
OUT_DIR = Path(__file__).resolve().parents[1] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_nuclear_capacity_by_busbar(scenario: ScenarioResults) -> dict:
    """Extract new nuclear capacity from market steps by finding entries with price=9."""
    nuclear_cap = {}

    try:
        model = scenario.model
        for busbar_name in NO_BUSBARS:
            # Find all market steps for this busbar
            for market_step in model.market_steps():
                if market_step.busbar_name == busbar_name:
                    # Get price - handle both scalar and array cases
                    price = getattr(market_step.price, "scenarios", market_step.price)
                    if isinstance(price, (list, np.ndarray)):
                        price = price[0] if len(price) > 0 else None

                    # If price is 9, this is new nuclear
                    if price is not None and abs(price - NUCLEAR_MARGINAL_COST) < 0.1:
                        capacity = getattr(market_step.capacity, "scenarios", market_step.capacity)
                        if isinstance(capacity, (list, np.ndarray)):
                            capacity = capacity[0] if len(capacity) > 0 else 0

                        if capacity > 0:
                            nuclear_cap[busbar_name] = {
                                "capacity": capacity,
                                "name": market_step.name,
                                "price": price,
                            }
    except Exception as e:
        print(f"Warning: Could not extract nuclear capacity: {e}")

    return nuclear_cap


def analyze_nuclear_dispatch(scenario_results: dict, output_dir: Path):
    """Analyze new nuclear dispatch, capacity factor, and value factor."""

    print("\n" + "=" * 70)
    print("NEW NUCLEAR DISPATCH ANALYSIS")
    print("=" * 70)
    print(f"Marginal cost threshold: {NUCLEAR_MARGINAL_COST} €/MWh")
    print("Dispatch rule: price > MC → full production, price <= MC → zero production\n")

    results = {}

    for scenario_name, scenario in scenario_results.items():
        print(f"Analyzing {scenario_name}...")

        try:
            scenario_results_data = {}

            # Get nuclear capacity for each busbar
            nuclear_cap = get_nuclear_capacity_by_busbar(scenario)

            if not nuclear_cap:
                print("  No new nuclear found (price=9)")
                continue

            print(f"  Found new nuclear in: {', '.join(nuclear_cap.keys())}")

            # For each busbar with nuclear, compute dispatch
            for busbar_name, nuc_info in nuclear_cap.items():
                try:
                    # Get hourly prices
                    df_prices = scenario.get_prices_for_busbar(busbar_name)
                    prices = df_prices.mean(axis=1).values

                    # Compute dispatch: 1 if price > MC, 0 otherwise
                    dispatch = np.where(prices > NUCLEAR_MARGINAL_COST, 1.0, 0.0)

                    # Scale by capacity
                    dispatch_mw = dispatch * nuc_info["capacity"]

                    # Capacity factor
                    capacity_factor = np.mean(dispatch)

                    # Value factor: avg price when producing / MC
                    producing_hours = dispatch > 0.5
                    if producing_hours.any():
                        avg_price_when_producing = prices[producing_hours].mean()
                        value_factor = avg_price_when_producing / NUCLEAR_MARGINAL_COST
                    else:
                        avg_price_when_producing = 0.0
                        value_factor = 0.0

                    # Total energy produced
                    total_energy = dispatch_mw.sum()

                    # Revenue (energy * price when produced)
                    revenue = (dispatch_mw * prices).sum()

                    scenario_results_data[busbar_name] = {
                        "capacity": nuc_info["capacity"],
                        "dispatch": dispatch_mw,
                        "prices": prices,
                        "capacity_factor": capacity_factor,
                        "value_factor": value_factor,
                        "avg_price_when_producing": avg_price_when_producing,
                        "total_energy_mwh": total_energy,
                        "revenue": revenue,
                        "producing_hours": producing_hours.sum(),
                        "total_hours": len(prices),
                    }

                    # Print summary
                    print(f"    {busbar_name}:")
                    print(f"      Capacity: {nuc_info['capacity']:.1f} MW")
                    print(f"      Capacity factor: {capacity_factor:.2%}")
                    print(f"      Value factor: {value_factor:.3f}")
                    print(f"      Avg price when producing: {avg_price_when_producing:.2f} €/MWh")
                    print(f"      Total energy: {total_energy:.0f} MWh")
                    print(f"      Revenue: {revenue:.0f} €")
                    print(f"      Producing hours: {producing_hours.sum()}/{len(prices)}")

                except Exception as e:
                    print(f"    Error for {busbar_name}: {e}")

            results[scenario_name] = scenario_results_data

        except Exception as e:
            print(f"  Error: {e}")

    return results


def plot_nuclear_dispatch(results: dict, output_dir: Path):
    """Plot nuclear dispatch over time for each scenario."""

    if not results:
        print("No results to plot")
        return

    # Get all unique busbars across all scenarios
    all_busbars = set()
    for scenario_data in results.values():
        all_busbars.update(scenario_data.keys())
    all_busbars = sorted(all_busbars)

    # Create subplots: one per busbar
    fig, axes = plt.subplots(len(all_busbars), 1, figsize=(14, 3 * len(all_busbars)))
    if len(all_busbars) == 1:
        axes = [axes]

    for idx, busbar in enumerate(all_busbars):
        ax = axes[idx]

        for scenario_name, scenario_data in results.items():
            if busbar in scenario_data:
                data = scenario_data[busbar]
                dispatch = data["dispatch"]
                # Plot dispatch (show first 168 hours = 1 week for clarity)
                ax.plot(dispatch[:168], label=scenario_name, linewidth=1.5, alpha=0.8)

        ax.set_ylabel("Nuclear dispatch (MW)")
        ax.set_title(f"New Nuclear Dispatch - {busbar} (first week)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Hour")
    fig.tight_layout()

    # Save
    png_out = output_dir / "nuclear_dispatch.png"
    pdf_out = output_dir / "nuclear_dispatch.pdf"
    fig.savefig(png_out, dpi=150)
    fig.savefig(pdf_out)
    print(f"Saved: {png_out}")


def plot_nuclear_metrics(results: dict, output_dir: Path):
    """Create comparison plots of nuclear metrics (CF, VF) across scenarios."""

    if not results:
        return

    # Collect metrics
    metrics_data = []
    for scenario_name, scenario_data in results.items():
        for busbar, data in scenario_data.items():
            metrics_data.append(
                {
                    "Scenario": scenario_name,
                    "Busbar": busbar,
                    "Capacity Factor": data["capacity_factor"],
                    "Value Factor": data["value_factor"],
                }
            )

    df_metrics = pd.DataFrame(metrics_data)

    if df_metrics.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Capacity Factor
    ax = axes[0]
    df_cf = df_metrics.pivot(index="Busbar", columns="Scenario", values="Capacity Factor")
    df_cf.plot(kind="bar", ax=ax, color="steelblue", alpha=0.7)
    ax.set_ylabel("Capacity Factor")
    ax.set_title("New Nuclear Capacity Factor by Area")
    ax.set_xlabel("Area")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim([0, 1])

    # Value Factor
    ax = axes[1]
    df_vf = df_metrics.pivot(index="Busbar", columns="Scenario", values="Value Factor")
    df_vf.plot(kind="bar", ax=ax, color="coral", alpha=0.7)
    ax.axhline(y=1.0, color="r", linestyle="--", linewidth=1, label="MC reference")
    ax.set_ylabel("Value Factor")
    ax.set_title("New Nuclear Value Factor by Area")
    ax.set_xlabel("Area")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()

    # Save
    png_out = output_dir / "nuclear_metrics.png"
    pdf_out = output_dir / "nuclear_metrics.pdf"
    fig.savefig(png_out, dpi=150)
    fig.savefig(pdf_out)
    print(f"Saved: {png_out}")


if __name__ == "__main__":
    # Get model folder from command line or use default
    model_folder = sys.argv[1] if len(sys.argv) > 1 else "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_detFi_IncNOLoad"

    print(f"Using model folder: {model_folder}")

    base_path = Path.cwd()
    scenario_paths = find_scenario_results(base_path, model_folder=model_folder)

    if not scenario_paths:
        print(f"No scenarios found in {model_folder}")
        sys.exit(1)

    print(f"Found {len(scenario_paths)} scenarios: {list(scenario_paths.keys())}")

    # Load scenarios
    print("\nLoading scenario results...")
    scenario_results = {}
    for scenario_name, scenario_path in scenario_paths.items():
        try:
            print(f"  Loading {scenario_name}...")
            scenario_results[scenario_name] = ScenarioResults(scenario_path)
        except Exception as e:
            print(f"  Warning: Failed to load {scenario_name}: {e}")

    if not scenario_results:
        print("Failed to load any scenarios.")
        sys.exit(1)

    # Analyze and plot
    results = analyze_nuclear_dispatch(scenario_results, OUT_DIR)

    if results:
        plot_nuclear_dispatch(results, OUT_DIR)
        plot_nuclear_metrics(results, OUT_DIR)

        print("\n" + "=" * 70)
        print("Analysis complete!")
        print(f"Outputs saved to: {OUT_DIR}")
        print("=" * 70)
