import json

import pytest

from somerra_openadmet.challenge_submission import check_payload, payload, submit


def test_prediction_only_payload_has_no_link_or_code_metadata():
    fields = payload("/tmp/portal-upload/activity_submission.csv")
    assert fields[:3] == ["Jenhan", "Somerra", True]
    assert fields[3:8] == [""] * 5
    assert fields[8:11] == [False] * 3
    assert fields[11] == "Regression Prediction"
    assert fields[12]["orig_name"] == "activity_submission.csv"


@pytest.mark.parametrize(
    "index,value",
    [(7, "https://example.com/report"), (10, True), (8, True), (11, "TDI Prediction")],
)
def test_metadata_drift_fails_closed(index, value):
    fields = payload("/tmp/portal-upload/activity_submission.csv")
    fields[index] = value
    with pytest.raises(ValueError):
        check_payload(fields)


def test_remote_file_reference_is_forbidden():
    with pytest.raises(ValueError):
        payload("https://example.com/code.py")


def test_submission_requires_recorded_authorization_before_network(tmp_path, monkeypatch):
    from somerra_openadmet import challenge_submission

    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "completed"}))
    monkeypatch.setattr(
        challenge_submission,
        "json_request",
        lambda *args, **kwargs: pytest.fail("network called without authorization"),
    )
    with pytest.raises(ValueError, match="authorization"):
        submit(tmp_path, tmp_path)


@pytest.mark.parametrize("smiles_first", [False, True])
def test_submission_accepts_both_identifier_orders_before_preflight(
    tmp_path, monkeypatch, smiles_first
):
    import pandas as pd

    from somerra_openadmet import challenge_submission
    from somerra_openadmet.constants import ACTIVITY_ENDPOINTS, TEST_FILE
    from somerra_openadmet.download import sha256_file

    identifiers = pd.DataFrame(
        {"Molecule_Name": [f"mol{i}" for i in range(750)], "SMILES": ["CCO"] * 750}
    )
    identifiers.to_csv(tmp_path / TEST_FILE, index=False)
    frame = identifiers[["SMILES", "Molecule_Name"]] if smiles_first else identifiers.copy()
    for endpoint in ACTIVITY_ENDPOINTS:
        frame[endpoint] = [4.0 + i / 750 for i in range(750)]
    candidate = tmp_path / "activity_submission.csv"
    frame.to_csv(candidate, index=False)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "submission_authorized": True,
                "submission_sha256": sha256_file(candidate),
                "test_sha256": sha256_file(tmp_path / TEST_FILE),
                "open_source_code": False,
                "model_report_link": "",
            }
        )
    )

    def preflight():
        raise RuntimeError("validated locally; preflight reached")

    monkeypatch.setattr(challenge_submission, "live_preflight", preflight)
    monkeypatch.setattr(
        challenge_submission, "json_request", lambda *a, **k: pytest.fail("unexpected network")
    )
    with pytest.raises(RuntimeError, match="preflight reached"):
        submit(tmp_path, tmp_path)
    assert json.loads((tmp_path / "submission_receipt.json").read_text())["validation"]["valid"]
