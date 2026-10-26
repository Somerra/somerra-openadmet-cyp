import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.reference_member import Plateau, score_predictions


def test_plateau_tracks_material_gains_without_masking_small_checkpoint_gains():
    p = Plateau()
    assert not p.update(1, 1.0)
    for epoch in range(2, 31):
        assert not p.update(epoch, 0.999)
    assert p.update(31, 0.999)
    assert not p.update(32, 0.98)
    assert not p.update(61, 0.98)
    assert p.update(62, 0.98)
    with pytest.raises(ValueError, match="Nonfinite"):
        p.update(63, np.nan)


def test_scoring_preserves_observed_coverage_and_rejects_invalid_predictions():
    truth = pd.DataFrame()
    pred = pd.DataFrame()
    for ep in ACTIVITY_ENDPOINTS:
        truth[ep] = [3.0, 5.0, 8.0, np.nan]
        truth[ep + "_conf_low"] = [2.9, 4.9, 7.9, np.nan]
        truth[ep + "_conf_high"] = [3.1, 5.1, 8.1, np.nan]
        pred[ep] = [3.0, 5.0, 8.0, 0.0]
    result = score_predictions(truth, pred)
    assert result["MA-ST-RAE"] == 0
    assert all(c["labels"] == 3 for c in result["components"].values())
    with pytest.raises(ValueError, match="order"):
        score_predictions(truth, pred[pred.columns[::-1]])
    pred.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        score_predictions(truth, pred)


def test_reference_architecture_and_weighted_mae():
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from chemprop import nn
    from sklearn.preprocessing import StandardScaler

    from somerra_openadmet.reference_member import make_model

    scaler = StandardScaler().fit(np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]))
    model = make_model(scaler, list(ACTIVITY_ENDPOINTS))
    assert model.message_passing.output_dim == 2048
    assert model.init_lr == 0.0001 and model.max_lr == 0.001
    assert model.warmup_epochs == 2
    prediction = torch.tensor([[1.0, 3.0], [4.0, 2.0]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False], [True, True]])
    weights = torch.tensor([2.0, 0.5])
    loss = nn.metrics.MAE()(prediction, target, mask, weights)
    assert loss.item() == pytest.approx((2 + 2 + 1) / 3)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_targeted_task_selection_preserves_direct_endpoints():
    from somerra_openadmet.reference_member_data import select_tasks

    groups = {
        "single": ["single"],
        "tdi": ["tdi"],
        "chembl": ["chembl"],
        "pubchem_response": ["cyp3a4_max_response", "cyp2c19_max_response"],
        "pubchem_ac50": ["ac50"],
    }
    direct = list(ACTIVITY_ENDPOINTS)
    manifest = {"groups": groups, "tasks": direct + sum(groups.values(), [])}
    assert select_tasks(manifest, "official-only") == direct + ["single", "tdi"]
    assert select_tasks(manifest, "without-pubchem") == direct + ["single", "tdi", "chembl"]
    selected = select_tasks(manifest, "without-response-cyp3a4")
    assert "cyp3a4_max_response" not in selected
    assert "cyp2c19_max_response" in selected and "ac50" in selected
    assert selected[:4] == direct
    with pytest.raises(ValueError):
        select_tasks(manifest, "without-response-invalid")
    with pytest.raises(ValueError):
        select_tasks(manifest, "without-CYP3A4_pIC50_direct_inhibition")


def test_resume_copies_only_complete_checkpoint_curve_and_can_resume_again(tmp_path):
    import json

    from somerra_openadmet.reference_member import copy_recovery_curve

    previous, resumed, again = [tmp_path / name for name in ["previous", "resumed", "again"]]
    for path in [previous, resumed, again]:
        path.mkdir()
    rows = [{"epoch": i, "elapsed_seconds": i * 10} for i in [1, 2, 3]]
    (previous / "curve.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    copy_recovery_curve(previous, resumed, 2)
    assert [json.loads(x)["epoch"] for x in (resumed / "curve.jsonl").read_text().splitlines()] == [
        1,
        2,
    ]
    with (resumed / "curve.jsonl").open("a") as handle:
        handle.write(json.dumps(rows[2]) + "\n")
    copy_recovery_curve(resumed, again, 3)
    assert (again / "curve.jsonl").read_text() == (previous / "curve.jsonl").read_text()
    with pytest.raises(ValueError, match="incomplete"):
        copy_recovery_curve(previous, resumed, 4)
