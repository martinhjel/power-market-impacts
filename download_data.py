#!/usr/bin/env python3
"""
Download and extract the public Zenodo archives.

Run this once from the repo root before running any figure or table scripts:
    python download_data.py --record-id <zenodo_record_id>

Alternatively set:
    export ZENODO_RECORD_ID=<zenodo_record_id>

Requires: Python 3.8+, no third-party packages.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_RECORD_ID = "<record_id>"
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
IMPROVED_NUCLEAR_MODEL_FOLDER = f"{MODEL_FOLDER}_imp_nuke"

ARCHIVES = [
    {
        "filename": "data.tar.gz",
        "target": Path("data"),
        "label": "input data",
        "md5": "a5ab43a4fe25b958867dd7f0240fb40d",
    },
    {
        "filename": "results_processed.tar.gz",
        "target": Path("ltm_processed") / MODEL_FOLDER,
        "label": "processed simulation results",
        "md5": None,
    },
    {
        "filename": "results_processed_imp_nuke.tar.gz",
        "target": Path("ltm_processed") / IMPROVED_NUCLEAR_MODEL_FOLDER,
        "label": "processed simulation results with improved nuclear representation",
        "md5": "2a0187e9f26815d59b19d5167c9a64c0",
    },
]


def main() -> None:
    args = parse_args()
    record_id = args.record_id or os.environ.get("ZENODO_RECORD_ID") or DEFAULT_RECORD_ID

    if not record_id or record_id == DEFAULT_RECORD_ID:
        raise SystemExit(
            "Zenodo record ID is missing. Run with '--record-id <id>', "
            "set ZENODO_RECORD_ID, or update DEFAULT_RECORD_ID in download_data.py."
        )

    selected = set(args.only or [archive["filename"] for archive in ARCHIVES])
    unknown = selected - {archive["filename"] for archive in ARCHIVES}
    if unknown:
        raise SystemExit(f"Unknown archive(s) requested with --only: {', '.join(sorted(unknown))}")

    for archive in ARCHIVES:
        if archive["filename"] not in selected:
            continue
        download_and_extract(
            url=zenodo_file_url(record_id, archive["filename"]),
            archive=Path(archive["filename"]),
            target=archive["target"],
            label=archive["label"],
            md5=archive["md5"],
            force=args.force,
            keep_archive=args.keep_archives,
        )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-id",
        help="Zenodo record ID. Can also be set with the ZENODO_RECORD_ID environment variable.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[archive["filename"] for archive in ARCHIVES],
        help="Download only selected archive filenames.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and extract even when the target directory already exists.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .tar.gz archives after extraction.",
    )
    return parser.parse_args()


def zenodo_file_url(record_id: str, filename: str) -> str:
    quoted_filename = urllib.parse.quote(filename)
    return f"https://zenodo.org/records/{record_id}/files/{quoted_filename}?download=1"


def download_and_extract(
    url: str,
    archive: Path,
    target: Path,
    label: str,
    md5: Optional[str],
    force: bool,
    keep_archive: bool,
) -> None:
    if target.exists() and any(target.iterdir()) and not force:
        print(f"'{target}' already exists and is non-empty - skipping {label}.")
        return

    print(f"Downloading {label} ({archive.name}) from Zenodo...")
    urllib.request.urlretrieve(url, archive, reporthook=_progress)
    print()

    if md5:
        verify_md5(archive, md5)

    print(f"Extracting {archive} ...")
    extract_tar_gz(archive)
    normalize_extracted_target(target)
    if not target.exists():
        raise RuntimeError(
            f"Extraction completed, but expected '{target}' was not found. "
            "Check that the archive contains either that folder path or a top-level "
            f"'{target.name}' folder."
        )
    print(f"Done. Expected target: '{target}'.")

    if not keep_archive:
        archive.unlink()
        print(f"Removed {archive}.")


def verify_md5(path: Path, expected: str) -> None:
    actual = file_md5(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")
    print(f"Verified MD5: {actual}")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size <= 0:
        mb = downloaded / 1024**2
        print(f"\r  {mb:.0f} MB", end="", flush=True)
        return

    pct = min(downloaded / total_size * 100, 100)
    mb = downloaded / 1024**2
    total_mb = total_size / 1024**2
    print(f"\r  {pct:.1f}%  ({mb:.0f} / {total_mb:.0f} MB)", end="", flush=True)


def extract_tar_gz(archive: Path) -> None:
    if archive.suffixes[-2:] != [".tar", ".gz"] and archive.suffix != ".tgz":
        raise ValueError(f"Unsupported archive format: {archive}")

    with tarfile.open(archive, "r:gz") as tar:
        safe_extract(tar, Path("."))


def normalize_extracted_target(target: Path) -> None:
    """Accept archives rooted either at target or at target.name."""
    if target.exists():
        return

    root_name = Path(target.name)
    if not root_name.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    root_name.rename(target)


def safe_extract(tar: tarfile.TarFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if target_dir not in [member_path, *member_path.parents]:
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(target_dir)


if __name__ == "__main__":
    main()
