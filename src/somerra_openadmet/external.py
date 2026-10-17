"""Version-checked ChEMBL CYP assay snapshots with explicit censoring."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .assays import potency
from .campaign import parent_smiles, write_json
from .data import load_test
from .download import sha256_file

ROOT = "https://www.ebi.ac.uk/chembl/api/data/"


def request_json(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


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


def pubchem(data_dir: Path, output_dir: Path) -> dict:
    """Fetch AID 1851 outcomes, preserving inactive labels as classification targets."""
    root = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
    output_dir.mkdir(parents=True, exist_ok=False)
    raw = output_dir / "raw"
    raw.mkdir()
    url = root + "assay/aid/1851/concise/CSV"
    with urlopen(url, timeout=120) as response:
        content = response.read()
    (raw / "outcomes.csv").write_bytes(content)
    frame = pd.read_csv(io.BytesIO(content))
    frame = frame.loc[frame["CID"].notna()].copy()
    cids = sorted(frame.CID.astype(int).unique())
    molecules = {}
    for offset in range(0, len(cids), 500):
        payload = urlencode({"cid": ",".join(map(str, cids[offset : offset + 500]))}).encode()
        request = Request(root + "compound/cid/property/CanonicalSMILES/JSON", data=payload)
        result = request_json(request)
        write_json(raw / f"structures-{offset:06d}.json", result)
        for entry in result["PropertyTable"]["Properties"]:
            molecules[entry["CID"]] = entry.get("ConnectivitySMILES", entry.get("CanonicalSMILES"))
        print(json.dumps({"cid_offset": offset, "total": len(cids)}), flush=True)
        time.sleep(0.25)
    # Gene IDs are supplied explicitly in the PubChem panel export.
    isoforms = {1544: "1A2", 1557: "2C19", 1559: "2C9", 1565: "2D6", 1576: "3A4"}
    test_parents = set(load_test(data_dir).SMILES.map(parent_smiles))
    normalized = {}
    for cid, smiles in molecules.items():
        try:
            normalized[cid] = parent_smiles(smiles)
        except (ValueError, TypeError):
            continue
    records = []
    for row in frame.to_dict("records"):
        cid = int(row["CID"])
        gene = row["Target GeneID"]
        outcome = row["Activity Outcome"]
        if (
            cid not in normalized
            or normalized[cid] in test_parents
            or gene not in isoforms
            or outcome not in {"Active", "Inactive"}
        ):
            continue
        records.append(
            {
                "Molecule_Name": f"CID{cid}",
                "SMILES": molecules[cid],
                "parent": normalized[cid],
                "task": f"PubChem_CYP{isoforms[gene]}_active",
                "value": int(outcome == "Active"),
                "source": "PubChem_AID1851",
                "task_type": "binary",
                "SID": row["SID"],
                "CID": cid,
            }
        )
    data = pd.DataFrame(records).drop_duplicates(["parent", "task", "value"])
    conflicts = data.groupby(["parent", "task"]).value.transform("nunique") > 1
    data = data.loc[~conflicts]
    data.to_csv(output_dir / "assays.csv", index=False)
    manifest = {
        "source": url,
        "rows": len(data),
        "parents": data.parent.nunique(),
        "conflicting_rows_excluded": int(conflicts.sum()),
        "assays_sha256": sha256_file(output_dir / "assays.csv"),
        "raw_sha256": {p.name: sha256_file(p) for p in sorted(raw.iterdir())},
        "note": "Binary outcomes only; inactive compounds have no assigned potency.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
