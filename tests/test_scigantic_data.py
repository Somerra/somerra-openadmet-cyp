import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.configuration import compose_config
from somerra_openadmet.data_sources.scigantic import REVISION, validate_snapshot


def test_derivative_requires_pinned_provenance_and_auxiliary_only():
    frame = pd.DataFrame(
        {
            "source": ["scigantic_chembl"],
            "task": ["cyp1a2_pic50_chembl"],
            "parent": ["CCO"],
            "value": [5.0],
        }
    )
    evidence = {"source": "scigantic_chembl", "upstream_revision": REVISION}
    validate_snapshot(frame, evidence)
    with pytest.raises(ValueError, match="provenance"):
        validate_snapshot(frame, {**evidence, "upstream_revision": "unknown"})
    for task, value in [("CYP1A2_pIC50_direct_inhibition", 5), ("cyp1a2_pic50_chembl", np.inf)]:
        with pytest.raises(ValueError, match="labels"):
            validate_snapshot(frame.assign(task=task, value=value), evidence)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_snapshot(pd.concat([frame, frame]), evidence)


def test_derivative_selection_retains_distinct_source_paths(tmp_path):
    config, provenance = compose_config(
        {"data_sources": ["official_direct", "scigantic_chembl"]},
        {
            "prepared_dir": "official",
            "auxiliary_sources": {"scigantic_chembl": "chembl", "scigantic_pubchem": "pubchem"},
        },
        tmp_path,
    )
    assert config["auxiliary_dirs"] == ["chembl"]
    assert provenance["selected_data_sources"] == ["official_direct", "scigantic_chembl"]
