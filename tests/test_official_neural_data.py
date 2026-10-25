import numpy as np
import pandas as pd

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.data_sources.common import parent_smiles
from somerra_openadmet.data_sources.official import neural_training_frame


def test_neural_data_uses_same_parent_and_aggregation_with_auxiliary():
    fit = pd.DataFrame({"SMILES": ["CC[NH3+].[Cl-]", "CCN", "C[C@H](O)CC"]})
    fit["parent"] = fit.SMILES.map(parent_smiles)
    for task in ACTIVITY_ENDPOINTS:
        fit[task] = [1.0, 3.0, 5.0]
    aux = pd.DataFrame(
        {"parent": ["c1ccccc1", "CCCCCCCC"], "task": ["aux", "aux"], "value": [2.0, 9.0]}
    )
    direct = neural_training_frame(fit, None, ["CCCCCCCC"])
    combined = neural_training_frame(fit, aux, ["CCCCCCCC"])
    pd.testing.assert_frame_equal(direct, combined.loc[direct.index, direct.columns])
    np.testing.assert_array_equal(direct.loc["CCN"], 2.0)
    assert "C[C@H](O)CC" not in direct.index
    assert "c1ccccc1" in combined.index
    assert "CCCCCCCC" not in combined.index


def test_auxiliary_task_selection_keeps_bundle_intact():
    import pytest

    from somerra_openadmet.data_sources.official import select_auxiliary_tasks

    frame = pd.DataFrame({"task": ["single", "tdi"], "value": [1.0, 2.0]})
    selected = select_auxiliary_tasks(frame, ["single"])
    assert selected.task.tolist() == ["single"]
    assert len(frame) == 2
    with pytest.raises(ValueError, match="unavailable"):
        select_auxiliary_tasks(frame, ["missing"])
