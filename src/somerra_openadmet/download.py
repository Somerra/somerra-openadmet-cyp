"""Pinned, checksummed challenge-data acquisition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import snapshot_download

from .constants import ACTIVITY_DATA_FILES, DEFAULT_DATA_REVISION, HF_DATASET


@dataclass(frozen=True)
class DataVerification:
    valid: bool
    dataset: str | None
    revision: str | None
    files_checked: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def verify_activity_data(data_dir: Path) -> DataVerification:
    """Verify a downloaded snapshot against its local provenance manifest."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        return DataVerification(False, None, None, 0, (f"Missing manifest: {manifest_path}",))

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return DataVerification(False, None, None, 0, (f"Invalid manifest: {exc}",))

    dataset = manifest.get("dataset")
    revision = manifest.get("revision")
    recorded_files = manifest.get("files")
    errors: list[str] = []
    if dataset != HF_DATASET:
        errors.append(f"Manifest dataset is {dataset!r}; expected {HF_DATASET!r}")
    if not isinstance(revision, str) or len(revision) != 40:
        errors.append("Manifest revision is not a 40-character Git commit SHA")
    if not isinstance(recorded_files, dict):
        errors.append("Manifest files entry is missing or invalid")
        recorded_files = {}

    expected_names = set(ACTIVITY_DATA_FILES)
    recorded_names = set(recorded_files)
    if missing := sorted(expected_names - recorded_names):
        errors.append(f"Manifest is missing required files: {missing}")
    if unexpected := sorted(recorded_names - expected_names):
        errors.append(f"Manifest contains unexpected files: {unexpected}")

    checked = 0
    for name in sorted(expected_names & recorded_names):
        path = data_dir / name
        if not path.is_file():
            errors.append(f"Missing data file: {name}")
            continue
        expected = recorded_files[name]
        if not isinstance(expected, dict):
            errors.append(f"Invalid manifest entry for {name}")
            continue
        checked += 1
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_size != expected.get("bytes"):
            errors.append(
                f"Size mismatch for {name}: expected {expected.get('bytes')}, found {actual_size}"
            )
        if actual_sha256 != expected.get("sha256"):
            errors.append(f"SHA-256 mismatch for {name}")

    return DataVerification(
        not errors,
        dataset if isinstance(dataset, str) else None,
        revision if isinstance(revision, str) else None,
        checked,
        tuple(errors),
    )
