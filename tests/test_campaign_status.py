import json

from somerra_openadmet.campaign_status import status


def test_status_distinguishes_artifacts_from_live_processes(tmp_path):
    complete = tmp_path / "regression-complete"
    complete.mkdir()
    (complete / "metrics.json").write_text(json.dumps({"MA-ST-RAE": 0.8}))
    (complete / "run_manifest.json").write_text(json.dumps({"gpu_hours": 1.0}))
    partial = tmp_path / "regression-partial"
    partial.mkdir()
    (partial / "oof_predictions.csv").write_text("endpoint,y_pred\n")
    (tmp_path / "regression-gpu-budget.json").write_text(
        json.dumps(
            {
                str(complete.resolve()): {"used_hours": 1.0, "reserved_hours": 2.0},
                str(partial.resolve()): {"reserved_hours": 2.0},
            }
        )
    )
    result = status(tmp_path)
    assert result["recorded_gpu_hours"] == 1.0
    assert result["outstanding_reserved_gpu_hours"] == 2.0
    assert [r["artifact_state"] for r in result["runs"]] == ["complete", "partial"]


def test_status_includes_reserved_run_before_first_fold_finishes(tmp_path):
    run = tmp_path / "regression-starting"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "gpu_hours": None,
                "selected_folds": [4],
                "completed_folds": [],
                "requested_folds_complete": False,
                "complete": False,
            }
        )
    )
    (tmp_path / "regression-gpu-budget.json").write_text(
        json.dumps(
            {
                str(run.resolve()): {"reserved_hours": 1.0, "status": "running"},
            }
        )
    )
    result = status(tmp_path)
    assert result["runs"][0]["selected_folds"] == [4]
    assert result["runs"][0]["completed_folds"] == []
    assert result["runs"][0]["artifact_state"] == "partial"
