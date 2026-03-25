from typing import List

import h5py


def list_hdf5_datasets(h5file: h5py.File) -> List[str]:
    """
    Recursively list all dataset paths in the HDF5 file.

    :param h5file: Opened h5py.File object.
    :return: List of dataset paths.
    """
    dataset_paths = []

    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            dataset_paths.append(name)

    h5file.visititems(visitor_func)
    return dataset_paths


def read_hdf5_dataset(h5file: h5py.File, path: str):
    """
    Reads a dataset from an HDF5 file.

    :param h5file: Opened h5py.File object.
    :param path: Dataset path inside HDF5.
    :return: Dataset as a NumPy array.
    """
    return h5file[path][:]


def build_hdf5_structure(h5file: h5py.File) -> dict[str, None]:
    """
    Recursively builds a nested dict representing the HDF5 file structure.

    :param h5file: Opened h5py.File
    :return: Nested dictionary of groups and datasets
    """
    def visit_group(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = visit_group(item)
            elif isinstance(item, h5py.Dataset):
                result[key] = item
        return result

    return visit_group(h5file)