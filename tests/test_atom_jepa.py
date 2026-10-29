import json

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.atom_jepa import align_rows, generate_conformer, pooled_views
from somerra_openadmet.representations import identities


def test_conformers_reproducible_with_hydrogens_and_stereo():
    z, xyz, status = generate_conformer("C[C@H](O)F")
    z2, xyz2, status2 = generate_conformer("C[C@H](O)F")
    np.testing.assert_array_equal(z, z2)
    np.testing.assert_array_equal(xyz, xyz2)
    assert status == status2 and 1 in z
    _, other, _ = generate_conformer("C[C@@H](O)F")
    assert not np.array_equal(xyz, other)


@pytest.mark.parametrize("smiles", ["not_smiles", "", "CC.[Na+]"])
def test_invalid_geometry_inputs_fail(smiles):
    with pytest.raises(ValueError):
        generate_conformer(smiles)


def test_atom_norm_precedes_pool_and_preserves_gradients():
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, 9, 256)
    x[0, 1] = 1
    x[1, 1] = -1
    x.requires_grad_(True)
    views = pooled_views(x, torch.tensor([0, 0]), 1)
    assert views["scalar"].shape == (1, 256)
    assert views["invariant"].shape == (1, 768)
    torch.testing.assert_close(views["invariant"][0, 256:512], torch.ones(256))
    views["invariant"].sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad)


def test_cached_row_alignment_preserves_ids_and_stereochemistry():
    rows = identities(
        pd.DataFrame({"Molecule_Name": ["a", "b"], "SMILES": ["C[C@H](O)F", "C[C@@H](O)F"]})
    )
    np.testing.assert_array_equal(align_rows(rows, rows.iloc[::-1]), [1, 0])
    query = rows.copy()
    query.loc[0, "SMILES"] = "C[C@@H](O)F"
    with pytest.raises(ValueError, match="SMILES"):
        align_rows(rows, query)
    with pytest.raises(ValueError):
        align_rows(rows, pd.concat([rows, rows]))


def test_conformer_cache_rejects_corruption(tmp_path):
    pytest.importorskip("torch")
    from somerra_openadmet.atom_jepa import GEOMETRY, load_conformers

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "complete": True,
                "geometry": GEOMETRY,
                "files": {"rows.csv": "wrong"},
            }
        )
    )
    (tmp_path / "rows.csv").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        load_conformers(tmp_path, pd.DataFrame())
