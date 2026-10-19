"""Version-checked ChEMBL CYP assay snapshots with explicit censoring."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from ..data import load_test
from ..download import sha256_file
from .common import parent_smiles, potency, request_json, write_json

ROOT = "https://www.ebi.ac.uk/chembl/api/data/"


def chembl(data_dir: Path, output_dir: Path, release: str = "ChEMBL_37") -> dict:
    status = request_json(ROOT + "status.json")
    if status["chembl_db_version"] != release:
        raise ValueError(f"Expected {release}, received {status['chembl_db_version']}")
    test_parents = set(load_test(data_dir).SMILES.map(parent_smiles))
    output_dir.mkdir(parents=True, exist_ok=False)
    raw = output_dir / "raw"
    raw.mkdir()
    write_json(raw / "status.json", status)
    records, exclusions = [], []
    parents = {}
    for isoform in ("1A2", "2C9", "2C19", "2D6", "3A4"):
        target_url = (
            ROOT
            + "target.json?"
            + urlencode({"pref_name": f"Cytochrome P450 {isoform}", "organism": "Homo sapiens"})
        )
        targets = request_json(target_url)
        write_json(raw / f"targets-{isoform}.json", targets)
        matches = [t for t in targets["targets"] if t["target_type"] == "SINGLE PROTEIN"]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous human CYP{isoform} target")
        target = matches[0]["target_chembl_id"]
        params = {
            "target_chembl_id": target,
            "standard_type": "IC50",
            "assay__confidence_score": 9,
            "limit": 1000,
            "offset": 0,
        }
        while True:
            page = request_json(ROOT + "activity.json?" + urlencode(params))
            write_json(raw / f"{isoform}-{params['offset']:06d}.json", page)
            for entry in page["activities"]:
                reason = None
                if entry.get("data_validity_comment") or entry.get("potential_duplicate"):
                    reason = "flagged_record"
                elif entry.get("target_organism") != "Homo sapiens":
                    reason = "nonhuman"
                elif entry.get("src_id") != 1:
                    # PubChem is ingested separately; avoid duplicating deposited qHTS records.
                    reason = "nonliterature_source"
                try:
                    value, relation = potency(
                        float(entry["standard_value"]),
                        entry["standard_units"],
                        entry["standard_relation"],
                    )
                    smiles = entry["canonical_smiles"]
                    if smiles not in parents:
                        parents[smiles] = parent_smiles(smiles)
                    parent = parents[smiles]
                    if parent in test_parents:
                        reason = "blinded_test_parent"
                except (ValueError, TypeError, KeyError):
                    reason = "invalid_structure_or_measurement"
                if reason:
                    exclusions.append({"activity_id": entry["activity_id"], "reason": reason})
                    continue
                records.append(
                    {
                        "Molecule_Name": entry["molecule_chembl_id"],
                        "SMILES": smiles,
                        "parent": parent,
                        "value": value,
                        "relation": relation,
                        "task": f"ChEMBL_CYP{isoform}_{entry['assay_chembl_id']}",
                        "source": release,
                        "activity_id": entry["activity_id"],
                        "assay_id": entry["assay_chembl_id"],
                        "document_id": entry["document_chembl_id"],
                        "target_id": target,
                    }
                )
            print(
                json.dumps(
                    {
                        "isoform": isoform,
                        "offset": params["offset"],
                        "total": page["page_meta"]["total_count"],
                    }
                ),
                flush=True,
            )
            if not page["page_meta"]["next"]:
                break
            params["offset"] += params["limit"]
    if request_json(ROOT + "status.json")["chembl_db_version"] != release:
        raise ValueError("ChEMBL release changed during download")
    data = pd.DataFrame(records).drop_duplicates("activity_id")
    data.to_csv(output_dir / "records.csv", index=False)
    pd.DataFrame(exclusions).to_csv(output_dir / "exclusions.csv", index=False)
    # Censored observations remain in records.csv; exact-label models must not use them as points.
    exact = data.loc[data.relation == "="].copy()
    exact.to_csv(output_dir / "assays.csv", index=False)
    manifest = {
        "source": ROOT,
        "release": release,
        "license": "CC BY-SA 3.0",
        "rows": len(data),
        "exact_rows": len(exact),
        "excluded": len(exclusions),
        "records_sha256": sha256_file(output_dir / "records.csv"),
        "assays_sha256": sha256_file(output_dir / "assays.csv"),
        "raw_sha256": {p.name: sha256_file(p) for p in sorted(raw.glob("*.json"))},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
