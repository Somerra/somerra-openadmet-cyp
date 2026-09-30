import json
from pathlib import Path

from somerra_openadmet.constants import ACTIVITY_DATA_FILES, HF_DATASET
from somerra_openadmet.download import sha256_file, verify_activity_data


def _write_snapshot(path: Path) -> None:
    path.mkdir()
    files = {}
    for name in ACTIVITY_DATA_FILES:
        target = path / name
        target.write_text(f"fixture for {name}\n")
        files[name] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    (path / "manifest.json").write_text(
        json.dumps({"dataset": HF_DATASET, "revision": "a" * 40, "files": files})
    )


def test_verify_activity_data_accepts_intact_snapshot(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_snapshot(data_dir)
    report = verify_activity_data(data_dir)
    assert report.valid
    assert report.files_checked == len(ACTIVITY_DATA_FILES)


def test_verify_activity_data_detects_tampering(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_snapshot(data_dir)
    (data_dir / ACTIVITY_DATA_FILES[0]).write_text("changed\n")
    report = verify_activity_data(data_dir)
    assert not report.valid
    assert any("mismatch" in error.lower() for error in report.errors)
