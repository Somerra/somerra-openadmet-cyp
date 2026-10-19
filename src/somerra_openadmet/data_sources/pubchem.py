"""PubChem CYP AID 1851 outcomes with binary activity labels."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from ..data import load_test
from ..download import sha256_file
from .common import parent_smiles, request_json, write_json


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
