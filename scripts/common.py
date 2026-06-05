"""
Common utilities and classes for EMPS visualization scripts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

try:
    from scripts.processed_results import ProcessedScenarioResults
except ModuleNotFoundError:
    from processed_results import ProcessedScenarioResults

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StyleMPL:
    color: str
    linestyle: str
    linewidth: float = 2.0
    marker: str | None = None
    markersize: float | None = None


class ScenarioStyler:
    """
    Assign unique colors and line styles to scenarios dynamically.
    Each scenario gets a distinct visual appearance based on its position in the list.
    """

    # Expanded color palette with distinct, visually appealing colors
    COLOR_PALETTE = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#aec7e8",  # light blue
        "#ffbb78",  # light orange
        "#98df8a",  # light green
        "#ff9896",  # light red
        "#c5b0d5",  # light purple
        "#c49c94",  # light brown
        "#f7b6d2",  # light pink
        "#c7c7c7",  # light gray
        "#dbdb8d",  # light olive
        "#9edae5",  # light cyan
    ]

    # Line styles for variety
    LINE_STYLES = [
        "solid",
        "dashed",
        "dashdot",
        "dotted",
    ]

    # Markers for additional distinction
    MARKERS = [
        None,
        "o",
        "s",
        "^",
        "v",
        "D",
        "*",
        "p",
    ]

    def __init__(self):
        self._scenario_index = {}
        self._counter = 0

    def _get_scenario_index(self, key: str) -> int:
        """Get or assign a unique index for each scenario."""
        if key not in self._scenario_index:
            self._scenario_index[key] = self._counter
            self._counter += 1
        return self._scenario_index[key]

    def color(self, key: str) -> str:
        """Assign color based on scenario index."""
        idx = self._get_scenario_index(key)
        return self.COLOR_PALETTE[idx % len(self.COLOR_PALETTE)]

    def mpl_style(self, key: str, width: float = 2.0) -> StyleMPL:
        """Assign complete style based on scenario index."""
        idx = self._get_scenario_index(key)

        # Assign color, line style, and marker cyclically
        color = self.COLOR_PALETTE[idx % len(self.COLOR_PALETTE)]
        linestyle = self.LINE_STYLES[(idx // len(self.COLOR_PALETTE)) % len(self.LINE_STYLES)]
        marker_idx = (idx // (len(self.COLOR_PALETTE) * len(self.LINE_STYLES))) % len(self.MARKERS)
        marker = self.MARKERS[marker_idx]

        # Check for UPRATE in name for additional visual distinction
        has_uprate = "_UPRATE_" in key or key.endswith("_UPRATE")

        return StyleMPL(
            color=color,
            linestyle=linestyle,
            linewidth=width + (0.3 if has_uprate else 0.0),
            marker=marker,
            markersize=5 if marker and has_uprate else (3 if marker else None),
        )


class ScenarioResults:
    """Load and cache processed EMPS simulation results."""

    def __init__(self, result_path: Path):
        self.result_path = Path(result_path)
        self.name = self.result_path.name
        self._processed = ProcessedScenarioResults.from_result_path(self.result_path)
        if self._processed is None:
            raise FileNotFoundError(
                f"Missing processed result data for {self.name}. Expected "
                "ltm_processed/<model>/<scenario>/processed_data.parquet."
            )
        logger.info(f"Using processed result data for {self.name}: {self._processed.data_path}")

    def get_busbars(self) -> Dict[str, any]:
        return {name: None for name in self.get_busbar_names()}
    
    def get_plants(self) -> Dict[str, any]:
        raise RuntimeError("Raw plant objects are not available; use processed result tables.")

    def get_prices_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._processed.get_prices_for_busbar(busbar_name)

    def get_hydro_production_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._processed.get_hydro_production_for_busbar(busbar_name)

    def get_reservoir_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._processed.get_reservoir_for_busbar(busbar_name)

    def get_load_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._processed.get_load_for_busbar(busbar_name)

    def get_market_steps_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        return self._processed.get_market_steps_for_busbar(busbar_name)

    def get_solar_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_solar_for_busbar(busbar_name)
        raise KeyError(f"Processed solar data not found for {busbar_name}")

    def get_onshore_wind_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_onshore_wind_for_busbar(busbar_name)
        raise KeyError(f"Processed onshore wind data not found for {busbar_name}")

    def get_offshore_wind_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_offshore_wind_for_busbar(busbar_name)
        raise KeyError(f"Processed offshore wind data not found for {busbar_name}")

    def get_fixed_nuclear_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_fixed_nuclear_for_busbar(busbar_name)
        raise KeyError(f"Processed fixed nuclear data not found for {busbar_name}")

    def get_historic_nuclear_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_historic_nuclear_for_busbar(busbar_name)
        raise KeyError(f"Processed historic nuclear data not found for {busbar_name}")

    def get_historic_nuclear_available_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_historic_nuclear_available_for_busbar(busbar_name)
        raise KeyError(f"Processed historic nuclear available data not found for {busbar_name}")

    def get_new_nuclear_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_new_nuclear_for_busbar(busbar_name)
        raise KeyError(f"Processed new nuclear data not found for {busbar_name}")

    def get_new_nuclear_available_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_new_nuclear_available_for_busbar(busbar_name)
        raise KeyError(f"Processed new nuclear available data not found for {busbar_name}")

    def get_total_nuclear_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_total_nuclear_for_busbar(busbar_name)
        raise KeyError(f"Processed total nuclear data not found for {busbar_name}")

    def get_total_nuclear_available_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_total_nuclear_available_for_busbar(busbar_name)
        raise KeyError(f"Processed total nuclear available data not found for {busbar_name}")

    def get_reservoir_spill_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_reservoir_spill_for_busbar(busbar_name)
        raise KeyError(f"Processed reservoir spill data not found for {busbar_name}")

    def get_reservoir_discharge_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_reservoir_discharge_for_busbar(busbar_name)
        raise KeyError(f"Processed reservoir discharge data not found for {busbar_name}")

    def get_reservoir_production(self, entity: str) -> pd.DataFrame:
        if self._processed is not None:
            return self._processed.get_reservoir_production(entity)
        raise KeyError(f"Processed reservoir production not found for {entity}")

    def get_busbar_names(self) -> list[str]:
        return self._processed.get_busbar_names()

    def get_dclines(self) -> Dict[str, any]:
        names = self._processed.get_dcline_names()
        return {name: None for name in names}

    def get_dcline_names(self) -> list[str]:
        return self._processed.get_dcline_names()

    def get_dcline_flow(self, dcline_name: str) -> pd.DataFrame:
        return self._processed.get_dcline_flow(dcline_name)


def add_grouped_legend(ax: plt.Axes, styler: ScenarioStyler):
    """Add simplified legend to plot showing only scenarios."""
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            title="Scenarios",
            loc="best",
            fontsize=8,
            framealpha=0.95,
            ncol=1 if len(handles) <= 6 else 2,
        )


def load_scenarios(scenario_paths: Dict[str, Path]) -> Dict[str, ScenarioResults]:
    """Load scenario results from paths."""
    scenario_results = {}
    for scenario_name, scenario_path in scenario_paths.items():
        try:
            logger.info(f"Loading {scenario_name}...")
            scenario_results[scenario_name] = ScenarioResults(scenario_path)
        except Exception as e:
            logger.warning(f"Failed to load {scenario_name}: {e}")
    return scenario_results
