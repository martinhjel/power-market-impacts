import streamlit as st
from app.utils.hdf5 import build_hdf5_structure
import h5py
import pandas as pd

def navigate_hdf5_structure(structure: dict[str, None]) -> list[str]:
    """
    Create dropdowns for each level in the nested HDF5 structure.

    :param structure: Nested dict from build_hdf5_structure
    :return: List of keys leading to the selected dataset
    """
    path = []
    current = structure

    while isinstance(current, dict):
        options = list(current.keys())
        if not options:
            break
        selected = st.selectbox(f"Select from {'.'.join(path) or 'root'}", options, key=">".join(path))
        path.append(selected)
        current = current[selected]

    return path

path = None
if "path" in st.session_state:
    st.write("Path:", st.session_state["path"])
    path = st.session_state["path"]
    
st.title("Hierarchical HDF5 Explorer")
if st.checkbox("Start HDF5 explorer", value=False):
    try:
        with h5py.File(path / "results/results.h5", "r") as h5file:
            structure = build_hdf5_structure(h5file)
            selected_path = navigate_hdf5_structure(structure)

            try:
                dataset = h5file["/".join(selected_path)][:]
                st.write(f"Shape: {dataset.shape}, Dtype: {dataset.dtype}")

                try:
                    df = pd.DataFrame(dataset)
                    st.dataframe(df)
                    st.line_chart(df)
                except Exception as e:
                    st.warning(f"Raw data preview (non-tabular): {e}")
                    st.write(dataset)
            except Exception as e:
                st.error(f"Invalid selection or non-dataset: {e}")
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        st.stop()
