from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.config import KAGGLE_DATASET_SLUG, RAW_DIR, RAW_FILE_NAME, ensure_dirs


def _copy_csv_from_folder(source_folder: Path) -> Path:
    candidates = sorted(source_folder.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in downloaded path: {source_folder}")

    # Prefer the canonical filename when available; otherwise use first CSV.
    preferred = [p for p in candidates if p.name.lower() == RAW_FILE_NAME.lower()]
    chosen = preferred[0] if preferred else candidates[0]

    target = RAW_DIR / RAW_FILE_NAME
    shutil.copyfile(chosen, target)
    return target


def download_with_kagglehub(dataset_slug: str = KAGGLE_DATASET_SLUG) -> Optional[Path]:
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None

    try:
        download_path = kagglehub.dataset_download(dataset_slug)
        return _copy_csv_from_folder(Path(download_path))
    except Exception:
        return None


def download_with_kaggle_cli(dataset_slug: str = KAGGLE_DATASET_SLUG) -> Optional[Path]:
    try:
        command = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(RAW_DIR),
            "--unzip",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return _copy_csv_from_folder(RAW_DIR)
    except Exception:
        return None


def download_dataset(dataset_slug: str = KAGGLE_DATASET_SLUG) -> Path:
    ensure_dirs()

    result = download_with_kagglehub(dataset_slug)
    if result is not None:
        return result

    result = download_with_kaggle_cli(dataset_slug)
    if result is not None:
        return result

    raise RuntimeError(
        "Kaggle download failed. Authenticate Kaggle first by placing kaggle.json in "
        "%USERPROFILE%/.kaggle/kaggle.json, then rerun download."
    )
