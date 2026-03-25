"""
Common utilities and classes for EMPS visualization scripts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

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
    """Load and cache EMPS simulation results."""

    def __init__(self, result_path: Path):
        self.result_path = Path(result_path)
        self.name = self.result_path.name
        self._session = None
        self._model = None

    @property
    def session(self):
        if self._session is None:
            from lpr_sintef_bifrost.ltm import LTM

            self._session = LTM.session_from_folder(self.result_path / "run_folder/emps")
        return self._session

    @property
    def model(self):
        if self._model is None:
            self._model = self.session.model
        return self._model

    def get_busbars(self) -> Dict[str, any]:
        return {b.name: b for b in self.model.busbars()}
    
    def get_plants(self) -> Dict[str, any]:
        return {p.name: p for p in self.model.plants()}

    def get_prices_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

        busbars = self.get_busbars()
        if busbar_name not in busbars:
            raise KeyError(f"Busbar {busbar_name} not found")
        return df_from_pyltm_result(busbars[busbar_name].market_result_price())

    def get_hydro_production_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

        busbars = self.get_busbars()
        if busbar_name not in busbars:
            raise KeyError(f"Busbar {busbar_name} not found")
        return df_from_pyltm_result(busbars[busbar_name].sum_hydro_production())

    def get_reservoir_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

        busbars = self.get_busbars()
        if busbar_name not in busbars:
            raise KeyError(f"Busbar {busbar_name} not found")
        return df_from_pyltm_result(busbars[busbar_name].sum_reservoir())

    def get_load_for_busbar(self, busbar_name: str) -> pd.DataFrame:
        from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

        busbars = self.get_busbars()
        if busbar_name not in busbars:
            raise KeyError(f"Busbar {busbar_name} not found")
        return df_from_pyltm_result(busbars[busbar_name].sum_load())

    def get_dclines(self) -> Dict[str, any]:
        return {dcline.name: dcline for dcline in self.model.dclines()}

    def get_dcline_flow(self, dcline_name: str) -> pd.DataFrame:
        from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

        dclines = self.get_dclines()
        if dcline_name not in dclines:
            raise KeyError(f"DC line {dcline_name} not found")
        return df_from_pyltm_result(dclines[dcline_name].transmission_results())


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
