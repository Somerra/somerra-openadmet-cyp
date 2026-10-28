import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.briford import calibrated, safe_batch_size
from somerra_openadmet.data_sources.workbench import (
    DIRECT,
    ISOFORMS,
    join_public,
    scaffold_folds,
    task_weights,
)


def test_historical_placement_preserves_order_and_sample_spread():
    raw = np.linspace(3.0, 7.0, 750)
    result, details = calibrated(raw, "cyp2d6", 0.45)
    assert result.mean() == pytest.approx(3.57)
    assert result.std(ddof=1) == pytest.approx(0.9)
    assert details["floor_count"] == 0
    assert np.array_equal(np.argsort(raw), np.argsort(result))
    a1, _ = calibrated(raw, "cyp1a2", 0.6)
    assert a1.std(ddof=1) == pytest.approx(0.6 * 1.32 * 1.553)
    assert a1.mean() == pytest.approx(4.412)


def test_invalid_calibration_does_not_silently_clip_correlation():
    with pytest.raises(ValueError):
        calibrated(np.ones(750), "cyp2d6", 0.4)
    with pytest.raises(ValueError):
        calibrated(np.arange(750), "cyp2d6", 0.9)


def test_primary_weighting_equalizes_observed_label_mass():
    frame = pd.DataFrame({c: [1.0] * (i + 1) + [np.nan] * (3 - i) for i, c in enumerate(DIRECT)})
    weights = task_weights(frame, "union18")
    masses = np.arange(1, 5) * weights[:4]
    assert np.allclose(masses, masses[0])
    assert len(weights) == 18
    assert np.allclose(weights[4:], 0.3)


def test_public_join_keeps_assay_scales_and_inactive_response():
    challenge = pd.DataFrame(
        {"molecule_name": ["official"], "smiles": ["CCO"], **{c: [4.0] for c in DIRECT}}
    )
    from somerra_openadmet.data_sources.workbench import connectivity

    chembl = pd.DataFrame(
        {
            "chembl_id": ["c1", "c2"],
            "smiles": ["CCO", "CCCl"],
            "inchi_key": [connectivity("CCO"), connectivity("CCCl")],
            **{f"{s}_pic50": [8.0, 2.0] for s in ISOFORMS + ["cyp2c19"]},
        }
    )
    veith = pd.DataFrame(
        [
            {"smiles": smi, "isoform": s, "max_response": value}
            for smi, value in [("CCO", 100), ("CCN", -200)]
            for s in ISOFORMS + ["cyp2c19"]
        ]
    )
    joined = join_public(challenge, chembl, veith).set_index("smiles")
    assert len(joined) == 3
    assert joined.loc["CCO", DIRECT[0]] == 4.0
    assert joined.loc["CCO", "cyp3a4_pic50_chembl"] == 8.0
    assert joined.loc["CCO", "cyp3a4_max_response"] == 50
    assert joined.loc["CCN", "cyp3a4_max_response"] == -150
    assert pd.isna(joined.loc["CCN", DIRECT[0]])


def test_scaffold_rotation_keeps_groups_together():
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "C1CCCCC1",
        "c1ccncc1",
        "c1ccoc1",
        "c1ccsc1",
        "C1CCC1",
        "CCO",
        "CCN",
    ]
    fold, scaffolds = scaffold_folds(smiles)
    assert set(fold) == set(range(5))
    assert np.array_equal(fold, scaffold_folds(smiles)[0])
    assert fold[0] == fold[1]
    assert fold[-1] == fold[-2]
    for s in set(scaffolds):
        assert len(set(fold[np.array(scaffolds) == s])) == 1


def test_prediction_batch_never_drops_singleton_tail():
    assert safe_batch_size(129) == 65
    assert safe_batch_size(128) == 64
