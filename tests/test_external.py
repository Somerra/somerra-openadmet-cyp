import pandas as pd
import pytest

from somerra_openadmet import external


def test_chembl_release_mismatch_stops_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "request_json", lambda _: {"chembl_db_version": "unexpected"})
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Expected"):
        external.chembl(tmp_path, output)
    assert not output.exists()


def test_chembl_preserves_censoring_and_excludes_test_parents(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "load_test", lambda _: pd.DataFrame({"SMILES": ["CCO"]}))

    def response(url):
        if "status.json" in url:
            return {"chembl_db_version": "ChEMBL_37"}
        if "target.json" in url:
            return {"targets": [{"target_type": "SINGLE PROTEIN", "target_chembl_id": "T1"}]}
        entries = []
        for i, (smiles, relation) in enumerate([("CCN", "="), ("CCC", ">"), ("CCO", "=")]):
            entries.append(
                {
                    "activity_id": i,
                    "target_organism": "Homo sapiens",
                    "src_id": 1,
                    "standard_value": "100",
                    "standard_units": "nM",
                    "standard_relation": relation,
                    "canonical_smiles": smiles,
                    "molecule_chembl_id": f"M{i}",
                    "assay_chembl_id": "A1",
                    "document_chembl_id": "D1",
                }
            )
        return {"activities": entries, "page_meta": {"total_count": 3, "next": None}}

    monkeypatch.setattr(external, "request_json", response)
    output = tmp_path / "output"
    result = external.chembl(tmp_path, output)
    assert result["rows"] == 2
    assert result["exact_rows"] == 1
    records = pd.read_csv(output / "records.csv")
    assert set(records.parent) == {"CCN", "CCC"}
    assert records.loc[records.parent == "CCC", "relation"].item() == "<"
    assert records.value.tolist() == [7.0, 7.0]
