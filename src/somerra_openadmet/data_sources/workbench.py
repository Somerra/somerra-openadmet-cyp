"""Reconstruct the historical briford members directly from public measurement tables."""

import importlib.util
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..download import sha256_file
from .common import write_json

REVISION = "8c72c7559471febaf0962fd551b4eaab8db3a136"
UNION_REVISION = "971f6a7f24d86e59bab219347c3e14c006ae4e35"
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
DIRECT = [f"{s}_pic50_direct_inhibition" for s in ISOFORMS]
LOG2FC = [f"{s}_log2fc" for s in ISOFORMS]
CHEMBL = [f"{s}_pic50_chembl" for s in ISOFORMS + ["cyp2c19"]]
VEITH = [f"{s}_max_response" for s in ISOFORMS + ["cyp2c19"]]
D6 = [
    "cyp2d6_pic50_direct_inhibition",
    "cyp2d6_log2fc",
    "cyp2d6_pic50_tdi_condition",
    "cyp2d6_emax_vs_pos_ctrl_direct_inhibition",
    "cyp2d6_pic50_chembl",
    "cyp2d6_max_response",
]
SOURCES = {
    "inhibition": "comp_chem/openadmet/cyp/training/inhibition",
    "single": "comp_chem/openadmet/cyp/training/single_concentration",
    "tdi": "comp_chem/openadmet/cyp/training/tdi",
    "emax": "comp_chem/openadmet/cyp/training/emax",
    "chembl": "comp_chem/chembl/cyp_inhibition/all_isoforms",
    "veith": "comp_chem/pubchem/cyp_inhibition/all_isoforms",
    "blind": "comp_chem/openadmet/cyp/testing/blinded",
}
MEMBER_TASKS = {
    "union18": DIRECT + LOG2FC + CHEMBL + VEITH,
    "mt_aux8": DIRECT + LOG2FC,
    "d6_isoform6": D6,
    "d6_single1": D6[:1],
}


def load_file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def connectivity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid molecular structure: {smiles}")
    return Chem.MolToInchiKey(mol).split("-")[0]


def scaffold_folds(smiles, seed=42):
    """Pinned Workbench: first-seen group IDs, randomly ordered greedy fold fill."""
    keys, group_ids, lookup = [], [], {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Invalid structure; random-split fallback is forbidden")
        key = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        keys.append(key)
        if key not in lookup:
            lookup[key] = len(lookup)
        group_ids.append(lookup[key])
    groups, counts = np.unique(group_ids, return_counts=True)
    if len(groups) < 5:
        raise ValueError("Need at least five scaffold groups")
    load = np.zeros(5, dtype=int)
    mapping = {}
    for i in np.random.default_rng(seed).permutation(len(groups)):
        k = int(np.argmin(load))
        mapping[groups[i]] = k
        load[k] += counts[i]
    return np.array([mapping[g] for g in group_ids]), keys


def task_weights(frame, member):
    tasks = MEMBER_TASKS[member]
    if member.startswith("d6"):
        return [1.0] + [0.3] * (len(tasks) - 1)
    counts = frame[DIRECT].notna().sum().to_numpy(dtype=np.float32)
    if np.any(counts <= 0):
        raise ValueError("Empty primary task")
    inv = 1 / counts
    primary = inv / inv.mean()
    return primary.tolist() + [float(0.3 * primary.mean())] * (len(tasks) - 4)


def join_public(challenge, chembl, veith):
    """Historical stable first-connectivity dedup and label-preserving left joins."""
    challenge = challenge.copy()
    challenge["key"] = challenge.smiles.map(connectivity)
    public_iso = ISOFORMS + ["cyp2c19"]
    chembl = chembl[
        ["chembl_id", "smiles", "inchi_key"] + [f"{s}_pic50" for s in public_iso]
    ].copy()
    chembl = chembl.rename(columns={f"{s}_pic50": f"{s}_pic50_chembl" for s in public_iso})
    chembl["key"] = chembl.inchi_key.str.split("-").str[0]
    chembl = chembl.drop(columns="inchi_key").dropna(subset="key").drop_duplicates("key")
    out = challenge.merge(chembl[["key"] + CHEMBL], on="key", how="left", validate="many_to_one")
    new = chembl.loc[~chembl.key.isin(challenge.key)].copy()
    new["molecule_name"] = new.chembl_id
    out = pd.concat([out, new.drop(columns="chembl_id")], ignore_index=True)
    wide = veith.pivot_table(index="smiles", columns="isoform", values="max_response")
    wide.columns = [f"{s}_max_response" for s in wide.columns]
    wide = wide[VEITH].clip(-150, 50).reset_index()
    wide["key"] = wide.smiles.map(connectivity)
    wide = wide.drop_duplicates("key")
    joined = out.merge(wide[["key"] + VEITH], on="key", how="left", validate="many_to_one")
    extra = wide.loc[~wide.key.isin(out.key)].copy()
    extra["molecule_name"] = "VEITH-" + extra.key
    return pd.concat([joined, extra], ignore_index=True).drop(columns="key")


def _download(item, root):
    name, key = item
    url = f"https://workbench-public-data.s3.us-west-2.amazonaws.com/{key}.csv"
    dest = root / "raw" / f"{name}.csv"
    with urllib.request.urlopen(url, timeout=90) as response:
        content = response.read()
        modified = response.headers.get("Last-Modified")
    dest.write_bytes(content)
    return name, {"url": url, "sha256": sha256_file(dest), "last_modified": modified}


def prepare(output, source, official_raw):
    output, source, official_raw = map(lambda p: Path(p).resolve(), (output, source, official_raw))
    if output.exists():
        raise FileExistsError(output)
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != REVISION or subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD"], text=True
    ):
        raise ValueError("Workbench source is not pristine and pinned")
    (output / "raw").mkdir(parents=True)
    (output / "source").mkdir()
    sources = dict(ThreadPoolExecutor(7).map(lambda item: _download(item, output), SOURCES.items()))
    evidence = {
        "reference_revision": REVISION,
        "union_revision": UNION_REVISION,
        "sources": sources,
        "status": "auditing",
        "submission_authorized": True,
        "open_source_code": False,
        "model_report_link": "",
    }
    write_json(output / "preparation.json", evidence)
    std_path = source / "src/workbench/utils/chem_utils/mol_standardize.py"
    shutil.copyfile(std_path, output / "source/mol_standardize.py")
    std = load_file_module(std_path, "briford_standardization")
    tables = {name: pd.read_csv(output / "raw" / f"{name}.csv") for name in SOURCES}
    official = tables["inhibition"]
    raw = pd.read_csv(official_raw / "cyp-challenge-TRAIN_inhibition.csv")
    raw = raw.set_index("Molecule_Name").reindex(official.molecule_name)
    for task in DIRECT:
        ep = task.split("_")[0].upper() + "_pIC50_direct_inhibition"
        for dest, orig in [
            (task, ep),
            (task + "_ci_lower", ep + "_conf_low"),
            (task + "_ci_upper", ep + "_conf_high"),
        ]:
            if not np.allclose(official[dest], raw[orig], rtol=0, atol=1e-7, equal_nan=True):
                raise ValueError(f"Official label mismatch: {task}")
    official.to_csv(output / "official.csv", index=False)
    # InferenceCache joins descriptors onto eval_df and preserves its input SMILES.
    # Do not replace them with the feature endpoint's standardized output.
    auxiliary = official.copy()
    single = tables["single"].pivot_table(
        index="molecule_name", columns="enzyme", values="log2fc_estimate"
    )
    single.columns = [f"{s.lower()}_log2fc" for s in single.columns]
    auxiliary = auxiliary.merge(
        single[LOG2FC], left_on="molecule_name", right_index=True, how="left", validate="one_to_one"
    )
    keep = ["molecule_name", "smiles"] + DIRECT + LOG2FC
    auxiliary = auxiliary[keep]
    union = join_public(auxiliary, tables["chembl"], tables["veith"])
    # Specialists use the expanded assay table; only the specialist's six heads train.
    tdi_cols = [f"{s}_pic50_tdi_condition" for s in ISOFORMS]
    emax_cols = [f"{s}_emax_vs_pos_ctrl_direct_inhibition" for s in ISOFORMS]
    side = tables["tdi"][["molecule_name", "smiles"] + tdi_cols].merge(
        tables["emax"][["molecule_name"] + emax_cols],
        on="molecule_name",
        how="left",
        validate="one_to_one",
    )
    expanded = auxiliary.merge(
        side.drop(columns="smiles"), on="molecule_name", how="left", validate="one_to_one"
    )
    expanded = pd.concat(
        [expanded, side.loc[~side.molecule_name.isin(auxiliary.molecule_name)]], ignore_index=True
    )
    expanded = join_public(expanded, tables["chembl"], tables["veith"])
    blind = pd.read_csv(official_raw / "cyp-challenge-TEST-BLINDED.csv")
    if set(blind.Molecule_Name) != set(tables["blind"].molecule_name) or len(blind) != 750:
        raise ValueError("Blind identities mismatch")
    blind.to_csv(output / "blind.csv", index=False)
    blind_keys = {
        connectivity(s)
        for s in std.standardize(
            blind.rename(columns={"SMILES": "smiles"}).copy(), extract_salts=True
        ).smiles
    }
    members, overlaps = {}, {}
    for name, tasks in MEMBER_TASKS.items():
        frame = auxiliary if name == "mt_aux8" else union if name == "union18" else expanded
        frame = (
            frame.loc[frame[tasks].notna().any(axis=1), ["molecule_name", "smiles"] + tasks]
            .copy()
            .reset_index(drop=True)
        )
        if frame.molecule_name.duplicated().any() or frame.smiles.isna().any():
            raise ValueError("Invalid member identities")
        frame["connectivity"] = frame.smiles.map(connectivity)
        overlap = frame.connectivity.isin(blind_keys)
        if frame.loc[overlap, [t for t in tasks if t in DIRECT]].notna().any(axis=None):
            raise ValueError("Blind parent has an official scored training label")
        overlaps[name] = frame.loc[overlap, ["molecule_name", "connectivity"]].to_dict("records")
        fold, scaffolds = scaffold_folds(frame.smiles)
        frame["fold"] = fold
        frame["scaffold"] = scaffolds
        # Exactly reproduce the pinned upstream grouping on this reconstructed row order.
        split_module = load_file_module(
            source / "src/workbench/training/splits.py", "briford_splits"
        )
        upstream = split_module.get_split_indices(
            frame, n_splits=5, strategy="scaffold", random_state=42
        )
        for k, (tr, va) in enumerate(upstream):
            if not np.array_equal(va, np.flatnonzero(fold == k)):
                raise ValueError("Fold parity failed")
            if frame.iloc[tr][tasks].notna().sum().eq(0).any():
                raise ValueError("Task has no training labels in a fold")
        frame.to_csv(output / f"{name}.csv", index=False)
        members[name] = {
            "tasks": tasks,
            "task_weights": task_weights(frame, name),
            "rows": len(frame),
            "labels": int(frame[tasks].notna().sum().sum()),
            "task_counts": frame[tasks].notna().sum().to_dict(),
            "fold_counts": frame.fold.value_counts().sort_index().to_dict(),
        }
    evidence.update(
        members=members,
        external_only_blind_overlap=overlaps,
        expected_counts={"official": 4905, "union18": 31670, "d6_single1": 1493},
        official_raw=str(official_raw),
        parity_scope="recipe_level_not_historical_weights",
        official_test_sha256=sha256_file(official_raw / "cyp-challenge-TEST-BLINDED.csv"),
    )
    write_json(output / "preparation.json", evidence)
    if (
        len(official) != 4905
        or members["union18"]["rows"] != 31670
        or members["d6_single1"]["rows"] != 1493
    ):
        raise ValueError(
            "Historical record counts differ; inspect preparation.json before training"
        )
    if members["union18"]["labels"] != 161400 or members["mt_aux8"]["labels"] != 24025:
        raise ValueError("Historical label counts differ; inspect preparation.json before training")
    evidence["file_sha256"] = {
        str(p.relative_to(output)): sha256_file(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "preparation.json"
    }
    evidence["status"] = "prepared"
    write_json(output / "manifest.json", evidence)
    return evidence
