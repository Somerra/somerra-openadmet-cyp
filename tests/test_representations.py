import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS, TEST_FILE
from somerra_openadmet.download import sha256_file
from somerra_openadmet.representations import (
    CHEMELEON_FEATURIZER,
    CHEMELEON_SHA256,
    Run,
    atomic_json,
    batch_indices,
    identities,
    load_cache,
    prepared_data,
    write_cache,
)


@pytest.fixture
def corpus(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    frame = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(30)],
            "SMILES": ["C" * (i + 1) for i in range(30)],
            "cluster": np.arange(30),
            "fold": np.arange(30) % 5,
        }
    )
    frame["parent"] = frame.SMILES
    for j, endpoint in enumerate(ACTIVITY_ENDPOINTS):
        frame[endpoint] = 4 + (np.arange(30) % 7) / 3 + j / 10
        frame[endpoint + "_conf_low"] = frame[endpoint] - 0.1
        frame[endpoint + "_conf_high"] = frame[endpoint] + 0.1
    frame.loc[0, ACTIVITY_ENDPOINTS[1]] = np.nan
    frame.to_csv(prepared / "training.csv", index=False)
    blind = pd.DataFrame({"Molecule_Name": [f"b{i}" for i in range(750)], "SMILES": "N"})
    blind_path = tmp_path / TEST_FILE
    blind.to_csv(blind_path, index=False)
    atomic_json(
        prepared / "manifest.json",
        {
            "training_sha256": sha256_file(prepared / "training.csv"),
            "n_folds": 5,
            "input_sha256": {TEST_FILE: sha256_file(blind_path)},
        },
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = identities(pd.concat([frame, blind]))
    matrix = np.random.default_rng(0).normal(size=(len(rows), 2048)).astype(np.float32)
    write_cache(
        cache,
        rows,
        matrix,
        {
            "encoder": "chemeleon",
            "representation": "mean",
            "featurizer": CHEMELEON_FEATURIZER,
            "checkpoint_sha256": CHEMELEON_SHA256,
        },
    )
    return frame, prepared, cache, matrix, blind_path


def test_cache_aligns_ids_and_does_not_bind_labels_or_fold_order(corpus):
    frame, _, cache, matrix, _ = corpus
    shuffled = frame.iloc[::-1].copy()
    shuffled["fold"] = 99
    shuffled[list(ACTIVITY_ENDPOINTS)] = 123
    loaded, _ = load_cache(cache, shuffled)
    np.testing.assert_array_equal(loaded, matrix[: len(frame)][::-1])


@pytest.mark.parametrize("fault", ["smiles", "duplicate", "missing"])
def test_cache_rejects_identity_errors(corpus, fault):
    frame, _, cache, _, _ = corpus
    query = frame.copy()
    if fault == "smiles":
        query.loc[0, "SMILES"] = "CCO"
    elif fault == "duplicate":
        query.loc[0, "Molecule_Name"] = query.loc[1, "Molecule_Name"]
    else:
        query.loc[0, "Molecule_Name"] = "not-cached"
    with pytest.raises(ValueError):
        load_cache(cache, query)


def test_cache_rejects_corruption_and_wrong_checkpoint(corpus):
    frame, _, cache, _, _ = corpus
    with pytest.raises(ValueError, match="provenance"):
        load_cache(cache, frame, checkpoint_sha256="different")
    with (cache / "embeddings.npy").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_cache(cache, frame)


def test_invalid_cache_never_commits_manifest(tmp_path):
    rows = identities(pd.DataFrame({"Molecule_Name": ["a"], "SMILES": ["CCO"]}))
    with pytest.raises(ValueError, match="array"):
        write_cache(tmp_path, rows, np.full((1, 2048), np.nan, dtype=np.float32), {})
    assert not (tmp_path / "manifest.json").exists()


def test_invalid_molecules_rejected_but_repeated_structures_preserved():
    frame = pd.DataFrame({"Molecule_Name": ["a", "b"], "SMILES": ["CCO", "CCO"]})
    assert len(identities(frame)) == 2
    frame.loc[0, "SMILES"] = "invalid_smiles"
    with pytest.raises(ValueError, match="Invalid molecular"):
        identities(frame)


def test_singleton_merge_keeps_every_training_row():
    batches = batch_indices(np.arange(9), 4, merge_singleton=True)
    assert [len(b) for b in batches] == [4, 5]
    np.testing.assert_array_equal(np.concatenate(batches), np.arange(9))
    assert [len(b) for b in batch_indices(np.arange(5), 4, merge_singleton=True)] == [5]
    assert [len(b) for b in batch_indices(np.arange(5), 4)] == [4, 1]
    with pytest.raises(ValueError, match="at least two"):
        batch_indices([0], 4, merge_singleton=True)


def test_prepared_rejects_split_leakage_even_with_updated_checksum(corpus):
    frame, prepared, _, _, _ = corpus
    frame.loc[1, "cluster"] = frame.loc[0, "cluster"]
    frame.to_csv(prepared / "training.csv", index=False)
    metadata = json.loads((prepared / "manifest.json").read_text())
    metadata["training_sha256"] = sha256_file(prepared / "training.csv")
    atomic_json(prepared / "manifest.json", metadata)
    with pytest.raises(ValueError, match="crosses folds"):
        prepared_data(prepared)


def test_interrupted_cpu_run_retains_incomplete_manifest(tmp_path):
    pytest.importorskip("torch")
    output = tmp_path / "run"
    with pytest.raises(TimeoutError):
        with Run({"output_dir": str(output), "device": "cpu", "max_gpu_hours": 0.01}) as run:
            run.check(projected_seconds=100)
    metadata = json.loads((output / "run_manifest.json").read_text())
    assert not metadata["complete"]
    assert metadata["gpu_hours"] == 0
    assert "TimeoutError" in metadata["error"]


def test_masked_mae_balances_heads_and_has_no_missing_gradients():
    torch = pytest.importorskip("torch")
    from somerra_openadmet.representation_training import masked_mae

    prediction = torch.zeros((2, 2), requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [1.0, 3.0]])
    loss = masked_mae(prediction, target)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert prediction.grad[0, 1] == 0
    assert torch.isfinite(prediction.grad).all()
    assert masked_mae(prediction, torch.full_like(target, float("nan"))) is None


def test_inner_holdout_is_group_disjoint_and_excludes_outer_labels(corpus):
    pytest.importorskip("torch")
    from somerra_openadmet.representation_training import inner_split

    frame, _, _, _, _ = corpus
    fit = np.flatnonzero(frame.fold != 0)
    train, val = inner_split(frame, fit, 42)
    assert not set(frame.iloc[train].cluster) & set(frame.iloc[val].cluster)
    assert set(train) | set(val) == set(fit)
    changed = frame.copy()
    changed.loc[changed.fold == 0, list(ACTIVITY_ENDPOINTS)] = 1e9
    train2, val2 = inner_split(changed, fit, 42)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(val, val2)


def test_scaled_learning_rates_are_bounded_and_reference_preserved():
    from somerra_openadmet.representation_training import recipe_for, scaled_recipes

    reference = recipe_for("full", {})
    candidates = scaled_recipes(reference, 1024)
    assert candidates[0] == reference
    assert {r["learning_rate"] for r in candidates} == {1e-4, 4e-4}
    assert all(r["warmup_fraction"] == 0.05 for r in candidates[1:])
    fixed = recipe_for("full", {"warmup_fraction": 0.05, "learning_rate": 4e-4})
    assert fixed["warmup_fraction"] == 0.05
    assert fixed["learning_rate"] == 4e-4


def test_full_model_gradients_update_encoder_and_head():
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet.representation_training import (
        Inputs,
        make_model,
        optimizer_for,
        recipe_for,
        step,
    )

    frame = pd.DataFrame({"SMILES": ["CCO", "CCN", "CC", "CO"]})
    inputs = Inputs(frame, "full", "cpu")
    torch.manual_seed(12)
    model = make_model("full", encoder_parameters={"d_h": 16}, input_dim=16)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    recipe = recipe_for("full", {})
    optimizer = optimizer_for(model, recipe)
    loss, _ = step(
        model, optimizer, inputs, np.arange(4), np.ones((4, 4), dtype=np.float32), "float32"
    )
    assert np.isfinite(loss)
    for prefix in ("encoder.", "head."):
        assert any(
            not torch.equal(v, before[k])
            for k, v in model.state_dict().items()
            if k.startswith(prefix)
        )


def test_frozen_five_fold_screen_and_blind_checkpoint_replay(corpus, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from somerra_openadmet import representation_training as training

    _, prepared, cache, _, blind = corpus

    # Frozen-head training must not instantiate an encoder.
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen training instantiated a foundation encoder")

    monkeypatch.setattr(training, "foundation", forbidden)
    config = {
        "prepared_dir": str(prepared),
        "cache_dir": str(cache),
        "output_dir": str(tmp_path / "screen"),
        "adaptation": "frozen",
        "device": "cpu",
        "max_gpu_hours": 0.1,
        "threads": 1,
        "epochs": 1,
        "batch_size": 8,
    }
    path = tmp_path / "screen.json"
    atomic_json(path, config)
    record = training.screen_representations(path)
    assert record["complete"]
    assert record["completed_folds"] == list(range(5))
    assert all(x <= 1e-4 for x in record["cpu_replay_max_abs"].values())
    oof = pd.read_csv(tmp_path / "screen/oof_predictions.csv")
    assert len(oof) == 119
    assert not oof.duplicated(["Molecule_Name", "endpoint"]).any()
    saved = torch.load(tmp_path / "screen/fold-0.pt", weights_only=True)
    assert not any(k.startswith("encoder.") for k in saved["state_dict"])
    for fold in range(5):
        evidence = json.loads((tmp_path / f"screen/fold-{fold}-training.json").read_text())
        assert not any(i % 5 == fold for i in evidence["training_indices"])
    export = {
        "run_dir": str(tmp_path / "screen"),
        "blind_csv": str(blind),
        "output_dir": str(tmp_path / "export"),
        "device": "cpu",
        "max_gpu_hours": 0.1,
        "threads": 1,
    }
    path = tmp_path / "export.json"
    atomic_json(path, export)
    result = training.predict_representations(path)
    assert result["validation"]["valid"]
    assert result["complete"] and not result["open_source_code"]
    assert result["model_report_link"] == ""
    assert len(pd.read_csv(tmp_path / "export/activity_submission.csv")) == 750


def test_h100_selection_resets_before_outer_refit_and_subset_is_incomplete(
    corpus, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    from somerra_openadmet import representation_training as training

    _, prepared, cache, _, _ = corpus
    monkeypatch.setattr(training, "profile_batches", lambda *a, **kw: 16)
    calls = []
    original = training.make_model

    def fresh(*args, **kwargs):
        model = original(*args, **kwargs)
        calls.append(model.head[0].weight.detach().clone())
        return model

    monkeypatch.setattr(training, "make_model", fresh)
    config = {
        "prepared_dir": str(prepared),
        "cache_dir": str(cache),
        "output_dir": str(tmp_path / "selected"),
        "adaptation": "frozen",
        "h100_select": True,
        "selected_folds": [0],
        "epochs": 1,
        "batch_size": 8,
        "device": "cpu",
        "max_gpu_hours": 0.1,
        "threads": 1,
    }
    path = tmp_path / "selected.json"
    atomic_json(path, config)
    result = training.screen_representations(path)
    assert result["requested_folds_complete"] and not result["complete"]
    assert not (tmp_path / "selected/metrics.json").exists()
    selection = json.loads((tmp_path / "selected/fold-0-selection.json").read_text())
    count = len(selection["trials"])
    # Every trial and the final refit have exactly the same fresh initialization.
    assert len(calls) == count + 2  # trials, final refit, checkpoint replay
    assert all(np.array_equal(calls[0], item) for item in calls[1 : count + 1])


def test_real_foundation_cache_matches_existing_cpu_encoder(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from chemprop import nn

    from somerra_openadmet.neural import loader
    from somerra_openadmet.representations import foundation, graph_batch, graph_dataset

    checkpoint = Path("runs/chemeleon_mp.pt")
    if not checkpoint.exists():
        pytest.skip("Local foundation checkpoint is not installed")
    torch.set_num_threads(2)
    model = foundation(checkpoint).eval()
    smiles = ["CCO", "C[C@H](N)C(=O)O", "c1ccccc1", "CC"]
    aggregate = nn.MeanAggregation()
    with torch.inference_mode():
        graph = graph_batch(graph_dataset(smiles), np.arange(len(smiles)), "cpu")
        actual = aggregate(model(graph), graph.batch).numpy()
        batch = next(iter(loader(smiles)))
        expected = aggregate(model(batch.bmg), batch.bmg.batch).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_real_cache_command_commits_training_and_blind_rows(corpus, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet.representations import cache_representations

    checkpoint = Path("runs/chemeleon_mp.pt")
    if not checkpoint.exists():
        pytest.skip("Local foundation checkpoint is not installed")
    frame, prepared, _, _, blind_path = corpus
    # Keep the extraction integration test small; this is not a challenge-data run.
    blind = pd.read_csv(blind_path).iloc[:2]
    blind.to_csv(blind_path, index=False)
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["input_sha256"][TEST_FILE] = sha256_file(blind_path)
    atomic_json(prepared / "manifest.json", manifest)
    output = tmp_path / "extracted"
    config = {
        "prepared_dir": str(prepared),
        "blind_csv": str(blind_path),
        "checkpoint": str(checkpoint),
        "output_dir": str(output),
        "device": "cpu",
        "max_gpu_hours": 0.1,
        "threads": 2,
        "batch_size": 8,
    }
    path = tmp_path / "extract.json"
    atomic_json(path, config)
    result = cache_representations(path)
    assert result["shape"] == [32, 2048]
    train_features, _ = load_cache(output, frame)
    blind_features, _ = load_cache(output, blind)
    assert train_features.shape == (30, 2048)
    assert blind_features.shape == (2, 2048)
    assert np.isfinite(blind_features).all()
    assert json.loads((output / "run_manifest.json").read_text())["complete"]


def test_full_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet.representation_training import (
        Inputs,
        make_model,
        optimizer_for,
        predictions,
        recipe_for,
        restore_checkpoint,
        save_checkpoint,
    )

    model = make_model("full", encoder_parameters={"d_h": 16}, input_dim=16)
    inputs = Inputs(pd.DataFrame({"SMILES": ["CCO", "CCN"]}), "full", "cpu")
    expected = predictions(model, inputs, np.arange(2), 2)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer_for(model, recipe_for("full", {})),
        {
            "adaptation": "full",
            "checkpoint_sha256": CHEMELEON_SHA256,
        },
    )
    restored, saved = restore_checkpoint(path)
    actual = predictions(restored, inputs, np.arange(2), 2)
    np.testing.assert_array_equal(expected, actual)
    assert saved["encoder_parameters"]["d_h"] == 16
    assert all(v.device == torch.device("cpu") for v in restored.state_dict().values())


def test_gpu_worker_keeps_reservation_for_supervisor_on_failure(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1 * 2**30, 96 * 2**30))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    output, ledger = tmp_path / "failed-gpu", tmp_path / "ledger.json"
    from somerra_openadmet.campaign_budget import CampaignLedger

    atomic_json(
        ledger, {"version": 1, "limit_hours": 12.0, "historical_hours": 0, "reservations": {}}
    )
    CampaignLedger(ledger).reserve_many([(output, 0.01)])
    monkeypatch.setenv("SOMERRA_CAMPAIGN_LEDGER", str(ledger))
    monkeypatch.setenv("SOMERRA_REPRESENTATION_WORKER", str(output.resolve()))
    with pytest.raises(RuntimeError, match="Insufficient free"):
        with Run(
            {
                "output_dir": str(output),
                "device": "cuda",
                "max_gpu_hours": 0.01,
            }
        ):
            pytest.fail("Should fail before entering the workload")
    records = json.loads(ledger.read_text())
    assert records["reservations"][str(output.resolve())]["status"] == "running"
    assert "used_hours" not in records["reservations"][str(output.resolve())]
    assert not json.loads((output / "run_manifest.json").read_text())["complete"]


def test_export_rejects_partial_runs_before_allocating(tmp_path):
    from somerra_openadmet.representation_training import predict_representations

    source = tmp_path / "partial"
    source.mkdir()
    atomic_json(source / "run_manifest.json", {"complete": False, "completed_folds": [0]})
    config = tmp_path / "export.json"
    output = tmp_path / "output"
    atomic_json(
        config,
        {
            "run_dir": str(source),
            "blind_csv": "unused.csv",
            "output_dir": str(output),
            "max_gpu_hours": 0.1,
            "device": "cpu",
        },
    )
    with pytest.raises(ValueError, match="completed five-fold"):
        predict_representations(config)
    assert not output.exists()


def test_shared_campaign_setting_wins_and_rejects_legacy(tmp_path, monkeypatch):
    from somerra_openadmet.representation_budget import ledger_path

    shared = tmp_path / "shared.json"
    atomic_json(
        shared, {"version": 1, "limit_hours": 12.0, "historical_hours": 0, "reservations": {}}
    )
    monkeypatch.setenv("SOMERRA_GPU_LEDGER", str(shared))
    monkeypatch.delenv("SOMERRA_CAMPAIGN_LEDGER", raising=False)
    assert ledger_path({"campaign_ledger": str(tmp_path / "other.json")}) == shared
    atomic_json(
        shared, {"version": 1, "limit_hours": 72, "historical_hours": 0, "reservations": {}}
    )
    with pytest.raises(ValueError, match="allocation contract"):
        ledger_path({})


def test_supervisor_reserves_before_launch_and_settles_after_exit(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from somerra_openadmet import representation_budget as budget
    from somerra_openadmet.campaign_budget import CampaignLedger

    ledger, output, config = tmp_path / "shared.json", tmp_path / "run", tmp_path / "config.json"
    atomic_json(
        ledger, {"version": 1, "limit_hours": 12.0, "historical_hours": 0, "reservations": {}}
    )
    atomic_json(config, {"output_dir": str(output), "device": "cuda", "max_gpu_hours": 0.01})
    monkeypatch.setenv("SOMERRA_CAMPAIGN_LEDGER", str(ledger))
    monkeypatch.delenv("SOMERRA_REPRESENTATION_WORKER", raising=False)

    class Child:
        returncode = 0

        def __init__(self, *args, **kwargs):
            assert CampaignLedger(ledger).snapshot()["reserved_hours"] == 0.01
            CampaignLedger(ledger).claim_worker(output, 0.01)
            output.mkdir()
            atomic_json(output / "worker-result.json", {"complete": True})

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(budget.subprocess, "Popen", Child)
    assert budget.maybe_supervise("screen", config)["complete"]
    record = json.loads(ledger.read_text())["reservations"][str(output)]
    assert record["status"] == "exited" and record["used_hours"] >= 0
    assert CampaignLedger(ledger).snapshot()["reserved_hours"] == 0


def test_standardization_is_training_only_and_checkpoint_replays(corpus, tmp_path):
    torch = pytest.importorskip("torch")
    from somerra_openadmet import representation_training as training

    frame, prepared, cache, _, _ = corpus
    config = {
        "prepared_dir": str(prepared),
        "cache_dir": str(cache),
        "output_dir": str(tmp_path / "normalized"),
        "adaptation": "frozen",
        "device": "cpu",
        "max_gpu_hours": 0.1,
        "threads": 1,
        "epochs": 1,
        "selected_folds": [0],
        "batch_size": 8,
        "head_dropout": 0.2,
        "standardize_targets": True,
    }
    path = tmp_path / "normalized.json"
    atomic_json(path, config)
    result = training.screen_representations(path)
    assert result["requested_folds_complete"]
    saved = torch.load(tmp_path / "normalized/fold-0.pt", weights_only=True)
    expected = frame.loc[frame.fold != 0, list(ACTIVITY_ENDPOINTS)].mean()
    np.testing.assert_allclose(saved["target_mean"], expected, rtol=1e-6)
    assert saved["head_dropout"] == 0.2
    assert result["cpu_replay_max_abs"]["0"] <= 1e-4


def test_cost_projection_does_not_extrapolate_gpu_startup_over_all_folds():
    from somerra_openadmet.representation_training import projected_training_seconds

    history = [{"seconds": 100}, {"seconds": 1}, {"seconds": 1}]
    assert projected_training_seconds(history, epochs=10, future_fits=4) == pytest.approx(56.4)
