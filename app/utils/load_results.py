from pathlib import Path

import streamlit as st
from lpr_sintef_bifrost.results import LTMSessionResults


@st.cache_data(show_spinner="Loading EMPS results...")
def load_results(path: Path) -> LTMSessionResults:
    """
    Load EMPS simulation results from a given path using the Bifrost API.

    Parameters
    ----------
    path : Path
        Path to the folder containing EMPS model output.

    Returns
    -------
    LTMSessionResults
        The parsed results object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Result folder does not exist: {path}")
    return LTMSessionResults.from_run_folder(path)
