"""Pinned, checksummed challenge-data acquisition."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import snapshot_download

from .constants import ACTIVITY_DATA_FILES, DEFAULT_DATA_REVISION, HF_DATASET


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_activity_data(
    output_dir: Path,
    revision: str = DEFAULT_DATA_REVISION,
) -> Path:
    """Download only the regression/TDI files and write a provenance manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(ACTIVITY_DATA_FILES),
        local_dir=output_dir,
    )

    missing = [name for name in ACTIVITY_DATA_FILES if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Dataset download incomplete; missing: {missing}")

    files = {
        name: {
            "sha256": sha256_file(output_dir / name),
            "bytes": (output_dir / name).stat().st_size,
        }
        for name in ACTIVITY_DATA_FILES
    }
    manifest = {
        "dataset": HF_DATASET,
        "revision": revision,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "scope": "activity_and_tdi_only",
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
