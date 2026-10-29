import importlib.util
import json
from pathlib import Path

import pytest

from somerra_openadmet.download import sha256_file

spec = importlib.util.spec_from_file_location(
    "jepa_campaign", Path(__file__).parents[1] / "scripts/run_mol_jepa_campaign.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resume_checks_artifacts_and_refuses_partial_output(tmp_path):
    output = tmp_path / "run"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"output_dir": str(output)}))
    assert module.completed_stage(config) is None
    output.mkdir()
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="Incomplete output"):
        module.completed_stage(config)
    artifact = output / "activity_submission.csv"
    artifact.write_text("predictions")
    record = {
        "complete": True,
        "config": {"output_dir": str(output)},
        "submission_sha256": sha256_file(artifact),
    }
    manifest.write_text(json.dumps(record))
    assert module.completed_stage(config) == record
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        module.completed_stage(config)
