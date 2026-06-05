#!/usr/bin/env python3
"""
Download and extract data archives from Zenodo.

Run this once from the repo root before running any figure or table scripts:
    python download_data.py

Requires: Python 3.8+, no third-party packages (uses stdlib only).
"""

import tarfile
import urllib.request
import zipfile
from pathlib import Path

# TODO: replace <record_id> with the actual Zenodo record ID after upload.
# Format: https://zenodo.org/records/<record_id>/files/<filename>
ARCHIVES = [
    {
        "url": "https://zenodo.org/records/<record_id>/files/data.zip",
        "archive": Path("data.zip"),
        "target": Path("data"),
        "label": "input data",
    },
    {
        "url": "https://zenodo.org/records/<record_id>/files/ltm_processed.tar.gz",
        "archive": Path("ltm_processed.tar.gz"),
        "target": Path("ltm_processed"),
        "label": "processed simulation results",
    },
    {
        "url": "https://zenodo.org/records/<record_id>/files/ltm_output.tar.gz",
        "archive": Path("ltm_output.tar.gz"),
        "target": Path("ltm_output"),
        "label": "LTM output metadata (config, dataset, logs)",
    },
]


def download_and_extract(url: str, archive: Path, target: Path, label: str) -> None:
    if target.exists() and any(target.iterdir()):
        print(f"'{target}' already exists and is non-empty — skipping {label}.")
        return

    print(f"Downloading {label} ({archive.name}) from Zenodo...")
    urllib.request.urlretrieve(url, archive, reporthook=_progress)
    print()

    print(f"Extracting {archive} ...")
    extract_archive(archive)
    print(f"Done. Extracted to '{target}'.")

    archive.unlink()
    print(f"Removed {archive}.")


def main() -> None:
    for entry in ARCHIVES:
        download_and_extract(**entry)
        print()


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1024 ** 2
        total_mb = total_size / 1024 ** 2
        print(f"\r  {pct:.1f}%  ({mb:.0f} / {total_mb:.0f} MB)", end="", flush=True)


def extract_archive(archive: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zip_archive:
            zip_archive.extractall(".")
        return

    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(".")
        return

    raise ValueError(f"Unsupported archive format: {archive}")


if __name__ == "__main__":
    main()
