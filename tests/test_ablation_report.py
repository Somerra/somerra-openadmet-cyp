import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_scigantic_ablation.py"


def test_report_preserves_existing_evidence_and_requires_explicit_overwrite(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"results": []}))
    summary = tmp_path / "summary.csv"
    summary.write_text("existing evidence")
    command = [sys.executable, str(SCRIPT), str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Output exists" in result.stderr
    assert summary.read_text() == "existing evidence"
    alternative = tmp_path / "new-summary.csv"
    subprocess.run(command + ["--output", str(alternative)], check=True, capture_output=True)
    assert alternative.exists()
    assert summary.read_text() == "existing evidence"
    subprocess.run(command + ["--overwrite"], check=True, capture_output=True)
    assert summary.read_text() != "existing evidence"
