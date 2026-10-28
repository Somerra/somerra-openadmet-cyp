import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.reference_benchmark import normalized_predictions


@pytest.mark.parametrize("suffix", ["", "_pred"])
def test_reference_endpoint_names_map_without_reordering_rows(suffix):
    frame = pd.DataFrame({"Molecule_Name": ["b", "a"]})
    for i, ep in enumerate(ACTIVITY_ENDPOINTS):
        frame[ep.lower() + suffix] = [i, i + 1]
    result = normalized_predictions(frame)
    assert result.Molecule_Name.tolist() == ["b", "a"]
    assert result.columns.tolist() == ["Molecule_Name", *ACTIVITY_ENDPOINTS]
    assert result[ACTIVITY_ENDPOINTS[2]].tolist() == [2, 3]


def test_reference_rejects_duplicate_ids_and_missing_endpoint():
    with pytest.raises(ValueError, match="Duplicate"):
        normalized_predictions(pd.DataFrame({"Molecule_Name": ["a", "a"]}))
    with pytest.raises(ValueError, match="Missing reference endpoint"):
        normalized_predictions(pd.DataFrame({"Molecule_Name": ["a"]}))
