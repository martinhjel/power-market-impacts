from pathlib import Path
import pandas as pd
from functools import lru_cache

class PowerGamaDataLoader:
    """
    Loads PowerGama input CSV files for a given dataset configuration.

    :param year: Dataset year (e.g., 2025)
    :param scenario: Scenario name (e.g., "BM")
    :param version: Dataset version string (e.g., "100")
    :param base_path: Base path to the dataset folder
    :param combined: (e.g. True)
    """

    def __init__(self, year: int, scenario: str, version: str, base_path: Path, combined: bool) -> None:
        self._generator: Optional[pd.DataFrame] = None  # Add backing field
        self.dataset_path = base_path / f"CASE_{year}/scenario_{scenario}/data/system"
        if combined:
            self.dataset_path = self.dataset_path / "combined"
        self.version = version
        self.scenario = scenario

        self.files = {
            "branch": self.dataset_path / f"branch_{scenario}_v{version}.csv",
            "generator": self.dataset_path / f"generator_{scenario}_v{version}.csv",
            "consumer": self.dataset_path / f"consumer_{scenario}_v{version}.csv",
            "dcbranch": self.dataset_path / f"dcbranch_{scenario}_v{version}.csv",
            "node": self.dataset_path / f"node_{scenario}_v{version}.csv",
        }

    def _validate(self, file: Path):
        if not file.exists():
            raise FileNotFoundError(file)

    @lru_cache(maxsize=None)
    def _read_file(self, file):
        return pd.read_csv(file, index_col=0)

    @property
    def branch(self) -> pd.DataFrame:
        self._validate(self.files["branch"])
        return self._read_file(self.files["branch"])

    @property
    def generator(self) -> pd.DataFrame:
        if self._generator is not None:
            return self._generator
        self._validate(self.files["generator"])
        return self._read_file(self.files["generator"])
    
    @generator.setter
    def generator(self, generator) -> None:
        self._generator = generator
        
    @property
    def consumer(self) -> pd.DataFrame:
        self._validate(self.files["consumer"])
        return self._read_file(self.files["consumer"])
        
    @property
    def dcbranch(self) -> pd.DataFrame:
        self._validate(self.files["dcbranch"])
        return self._read_file(self.files["dcbranch"])

    @property
    def node(self) -> pd.DataFrame:
        self._validate(self.files["node"])
        return self._read_file(self.files["node"])
