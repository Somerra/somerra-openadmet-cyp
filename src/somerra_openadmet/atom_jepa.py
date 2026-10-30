"""Pinned, frozen Atom-JEPA features for the matched representation benchmark.

Conformer coordinates are immutable inputs. Higher-order norms are taken BEFORE
atom pooling, matching the upstream ADMET implementation rather than embed().
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TEST_FILE
from .download import sha256_file
from .representations import Run, atomic_json, identities, prepared_data, provenance, write_cache

SOURCE_REVISION = "bab08533b0d8547fc51ca9a10e3e02c36444ca3b"
MODEL_REVISION = "8f7514f49693d836ca26dfe0bbae10880f75690b"
WEIGHTS_SHA256 = "0406152686fbae9f737f2c70c9acaf2da0eba6ad57e530142adfee47df75882f"
GEOMETRY = {
    "algorithm": "ETKDGv3",
    "seed": 42,
    "num_conformers": 1,
    "explicit_hydrogens": True,
    "enforce_chirality": True,
    "prune_rms": -1.0,
    "relaxation": "MMFF94",
    "max_iterations": 500,
    "retry": "random coordinates, same seed and chirality",
    "input": "original stereochemical SMILES; no parent standardization",
}
VIEWS = {"scalar": 256, "invariant": 768}


def corpus_rows(config):
    train, evidence = prepared_data(Path(config["prepared_dir"]))
    blind_path = Path(config["blind_csv"])
    if sha256_file(blind_path) != evidence["input_sha256"][TEST_FILE]:
        raise ValueError("Blind input differs from prepared manifest")
    blind = pd.read_csv(blind_path)
    rows = identities(pd.concat([train, blind], ignore_index=True))
    if set(rows.parent[: len(train)]) & set(rows.parent[len(train) :]):
        raise ValueError("Training and blind parents overlap")
    return rows, evidence


def generate_conformer(smiles, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Conformer seed must be a nonnegative int32")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("Invalid molecular SMILES")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Disconnected fragments require an explicitly reviewed policy")
    molecule = Chem.AddHs(molecule)
    retry = False
    for random_coordinates in (False, True):
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = seed
        parameters.enforceChirality = True
        parameters.pruneRmsThresh = -1.0
        parameters.useRandomCoords = random_coordinates
        parameters.numThreads = 1
        conformers = list(AllChem.EmbedMultipleConfs(molecule, numConfs=1, params=parameters))
        if len(conformers) == 1:
            retry = random_coordinates
            break
    else:
        raise ValueError("Conformer generation failed, including random-coordinate retry")
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError("Missing MMFF94 parameters")
    status, energy = AllChem.MMFFOptimizeMoleculeConfs(
        molecule, numThreads=1, maxIters=500, mmffVariant="MMFF94"
    )[0]
    coordinates = np.asarray(molecule.GetConformer(conformers[0]).GetPositions(), dtype=np.float32)
    if status not in (0, 1) or not np.isfinite(coordinates).all() or not np.isfinite(energy):
        raise ValueError("Invalid relaxed geometry")
    numbers = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64)
    if len(numbers) < 2 or not np.any(np.linalg.norm(coordinates - coordinates[0], axis=1) > 0.5):
        raise ValueError("Degenerate molecular geometry")
    return (
        numbers,
        coordinates,
        {
            "conformer_id": int(conformers[0]),
            "random_coordinate_retry": retry,
            "mmff_status": int(status),
            "energy": float(energy),
            "atoms": len(numbers),
        },
    )


def cache_conformers(config_path):
    from joblib import Parallel, delayed

    config = json.loads(Path(config_path).read_text())
    rows, evidence = corpus_rows(config)
    config = {**config, "device": "cpu", "max_gpu_hours": config.get("max_cpu_hours", 2)}
    with Run(config) as run:
        rows.to_csv(run.output / "rows.csv", index=False)
        results = []
        # Bounded chunks provide progress and deadline checks without task-order changes.
        for start in range(0, len(rows), 128):
            run.check()
            chunk = Parallel(n_jobs=config.get("workers", 8))(
                delayed(generate_conformer)(s) for s in rows.SMILES.iloc[start : start + 128]
            )
            results.extend(chunk)
            run.record["generated"] = len(results)
            run.save()
            print(f"conformers {len(results)}/{len(rows)}", flush=True)
        offsets = np.r_[0, np.cumsum([len(z) for z, _, _ in results])]
        temporary = run.output / "geometries.npz.partial"
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                atomic_numbers=np.concatenate([r[0] for r in results]),
                coordinates=np.concatenate([r[1] for r in results]),
                offsets=offsets,
            )
        os.replace(temporary, run.output / "geometries.npz")
        atomic_json(run.output / "relaxation.json", [r[2] for r in results])
        manifest = {
            "schema_version": 3,
            "kind": "atom_jepa_conformers",
            "complete": True,
            "geometry": GEOMETRY,
            "rows": len(rows),
            "input_sha256": evidence["input_sha256"],
            "files": {
                name: sha256_file(run.output / name)
                for name in ("rows.csv", "geometries.npz", "relaxation.json")
            },
            "nonconverged": sum(r[2]["mmff_status"] != 0 for r in results),
            "retried": sum(r[2]["random_coordinate_retry"] for r in results),
            **provenance(),
        }
        atomic_json(run.output / "manifest.json", manifest)
        run.record.update(complete=True, conformers=manifest)
        return manifest


def load_conformers(directory, frame):
    import torch

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 3
        or not manifest.get("complete")
        or manifest.get("geometry") != GEOMETRY
    ):
        raise ValueError("Incompatible or incomplete conformer cache")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Conformer checksum mismatch: {name}")
    rows = pd.read_csv(directory / "rows.csv")
    indices = align_rows(rows, frame)
    with np.load(directory / "geometries.npz", allow_pickle=False) as data:
        z, xyz, offsets = (data[k] for k in ("atomic_numbers", "coordinates", "offsets"))
    if (
        offsets.shape != (len(rows) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(z)
        or np.any(np.diff(offsets) < 2)
        or xyz.shape != (len(z), 3)
        or not np.isfinite(xyz).all()
        or np.any(z < 1)
        or np.any(z >= 128)
    ):
        raise ValueError("Invalid conformer dimensions or values")
    samples = [
        {
            "atomic_numbers": torch.tensor(z[offsets[i] : offsets[i + 1]]),
            "node_coordinates": torch.tensor(xyz[offsets[i] : offsets[i + 1]]),
        }
        for i in indices
    ]
    return samples, manifest


def align_rows(rows, frame):
    if not identities(rows).equals(rows):
        raise ValueError("Invalid cached row identities")
    query = identities(frame)
    positions = pd.Series(np.arange(len(rows)), index=rows.Molecule_Name)
    if not set(query.Molecule_Name).issubset(positions.index):
        raise ValueError("Cache missing molecule IDs")
    indices = positions.loc[query.Molecule_Name].to_numpy()
    if not np.array_equal(rows.iloc[indices].smiles_sha256, query.smiles_sha256):
        raise ValueError("Cache SMILES mismatch")
    return indices


def load_foundation(directory, device="cpu"):
    from atom_jepa import AtomJEPA

    directory = Path(directory)
    evidence = json.loads((directory / "provenance.json").read_text())
    if (
        evidence["source_revision"] != SOURCE_REVISION
        or evidence["model_revision"] != MODEL_REVISION
        or evidence["weights_license"] != "CC-BY-4.0"
        or evidence["files"].get("molecules/model.safetensors") != WEIGHTS_SHA256
    ):
        raise ValueError("Unreviewed Atom-JEPA checkpoint/license")
    for name, digest in evidence["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Model checksum mismatch: {name}")
    import atom_jepa

    source = Path(atom_jepa.__file__).parent
    for name, digest in evidence["source_files"].items():
        if sha256_file(source / name) != digest:
            raise ValueError(f"Source checksum mismatch: {name}")
    model = AtomJEPA.from_pretrained(
        str(directory / "molecules"),
        device=device,
        precision="fp32",
        cuequivariance=False,
        compile=False,
    )
    model.requires_grad_(False).eval()
    if model.embedding_dim != 256 or model.lmax != 2:
        raise ValueError("Unexpected pretrained representation dimensions")
    return model, evidence


def pooled_views(node_features, graph_index, num_graphs):
    """No learned normalization: return 256 scalars and 768 invariants per graph."""
    import torch

    if node_features.ndim != 3 or node_features.shape[1:] != (9, 256):
        raise ValueError("Expected per-atom [N,9,256] features")
    values = node_features.float()
    atom_features = torch.cat(
        [
            values[:, 0],
            (values[:, 1:4].square().sum(1) + 1e-8).sqrt(),
            (values[:, 4:9].square().sum(1) + 1e-8).sqrt(),
        ],
        dim=1,
    )
    pooled = torch.zeros(num_graphs, 768, dtype=torch.float32, device=values.device)
    pooled.index_add_(0, graph_index, atom_features)
    counts = torch.bincount(graph_index, minlength=num_graphs)
    if (counts == 0).any():
        raise ValueError("Empty molecular graph")
    pooled /= counts[:, None]
    if not torch.isfinite(pooled).all():
        raise ValueError("Non-finite Atom-JEPA representation")
    return {"scalar": pooled[:, :256], "invariant": pooled}


def encode(model, samples):
    batch = model.collate(samples)
    features = model(batch, degrees="all", per_atom=True)
    return pooled_views(features, batch["node_graph_index"].to(features.device), len(samples))


def cache_representations(config_path):
    import torch

    from .representation_budget import maybe_supervise

    supervised = maybe_supervise("atom-pilot-cache", config_path)
    if supervised is not None:
        return supervised

    config = json.loads(Path(config_path).read_text())
    rows, evidence = corpus_rows(config)
    samples, geometries = load_conformers(config["conformer_cache"], rows)
    with Run(config) as run:
        model, pin = load_foundation(config["checkpoint"], run.device)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        profile = []
        largest = sorted(range(len(samples)), key=lambda i: -len(samples[i]["atomic_numbers"]))
        with torch.inference_mode():
            for size in (4, 8, 16, 32):
                batch = [samples[i] for i in largest[:size]]
                try:
                    run.check()
                    encode(model, batch)
                    run.sync()
                    start = time.monotonic()
                    for _ in range(3):
                        encode(model, batch)
                    run.sync()
                    profile.append(
                        {
                            "batch_size": size,
                            "status": "ok",
                            "molecules_per_second": 3 * len(batch) / (time.monotonic() - start),
                        }
                    )
                except torch.cuda.OutOfMemoryError:
                    profile.append({"batch_size": size, "status": "oom"})
                    torch.cuda.empty_cache()
                atomic_json(run.output / "extraction-profile.json", profile)
            good = [p for p in profile if p["status"] == "ok"]
            if not good:
                raise RuntimeError("No extraction batch fits")
            peak = max(p["molecules_per_second"] for p in good)
            size = min(p["batch_size"] for p in good if p["molecules_per_second"] >= 0.95 * peak)
            run.check(len(rows) / peak * 1.25)
            matrices = {view: [] for view in VIEWS}
            started = time.monotonic()
            for i in range(0, len(samples), size):
                run.check()
                views = encode(model, samples[i : i + size])
                for name, value in views.items():
                    matrices[name].append(value.cpu().numpy())
                if i % (size * 20) == 0:
                    print(f"Atom-JEPA features {i + len(views['scalar'])}/{len(rows)}", flush=True)
                    run.record["extracted_rows"] = i + len(views["scalar"])
                    run.save()
            matrices = {k: np.concatenate(v).astype(np.float32) for k, v in matrices.items()}
            # Extraction cannot update either learned weights or running buffers.
            if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
                raise ValueError("Frozen encoder state changed")
            probe_ids = [0, largest[0], len(rows) - 1]
            probe = [samples[i] for i in probe_ids]
            online = encode(model, probe)
            parity = {}
            for name in VIEWS:
                cached = torch.tensor(matrices[name][probe_ids], device=run.device)
                parity[name] = float((cached - online[name]).abs().max())
                if not torch.allclose(cached, online[name], atol=1e-4, rtol=1e-4):
                    raise ValueError("Cached/online embedding parity failed")
            rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            rotated = [
                {**s, "node_coordinates": s["node_coordinates"] @ rotation + 2.0} for s in probe
            ]
            transformed = encode(model, rotated)
            invariance = {k: float((online[k] - transformed[k]).abs().max()) for k in VIEWS}
            relative_invariance = {
                k: float(
                    torch.linalg.vector_norm(online[k] - transformed[k])
                    / torch.linalg.vector_norm(online[k]).clamp_min(1e-12)
                )
                for k in VIEWS
            }
            # The unmodified upstream CPU encoder has the same ~1e-3 absolute
            # rotation residual (relative L2 ~2e-4). This is a model diagnostic,
            # distinct from cache/replay identity. See the retained CPU/GPU audit.
            if any(invariance[k] > 0.005 or relative_invariance[k] > 0.001 for k in VIEWS):
                raise ValueError(f"Rotation/translation invariance failed: {invariance}")
        root = {
            "schema_version": 3,
            "encoder": "atom_jepa",
            "complete": True,
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": pin["files"]["molecules/model.safetensors"],
            "precision": "float32",
            "numerical_tolerances": {
                "cache_online_atol": 1e-4,
                "cache_online_rtol": 1e-4,
                "rotation_max_abs": 0.005,
                "rotation_relative_l2": 0.001,
                "basis": "upstream CPU/GPU audit before evaluating any CYP scores",
            },
            "conformers_per_molecule": 1,
            "pooling": "per-atom l=1,2 sqrt(sum_m(x^2)+1e-8); concatenate l=0; atom mean",
            "geometry_manifest_sha256": sha256_file(
                Path(config["conformer_cache"]) / "manifest.json"
            ),
            "geometry": geometries,
            "prepared_manifest": evidence,
            "views": {},
            **provenance(),
        }
        for name, matrix in matrices.items():
            folder = run.output / name
            folder.mkdir()
            root["views"][name] = write_cache(
                folder,
                rows,
                matrix,
                {
                    "encoder": "atom_jepa",
                    "representation": name,
                    "checkpoint_sha256": root["checkpoint_sha256"],
                },
            )
        atomic_json(run.output / "manifest.json", root)
        run.record.update(
            complete=True,
            batch_size=size,
            rows=len(rows),
            frozen_state_unchanged=True,
            cached_online_max_abs=parity,
            invariance_max_abs=invariance,
            invariance_relative_l2=relative_invariance,
            molecules_per_second=len(rows) / (time.monotonic() - started),
        )
        return root


def load_view(directory, frame, view):
    directory = Path(directory)
    root = json.loads((directory / "manifest.json").read_text())
    if (
        root.get("schema_version") != 3
        or not root.get("complete")
        or root.get("encoder") != "atom_jepa"
        or root.get("source_revision") != SOURCE_REVISION
        or root.get("model_revision") != MODEL_REVISION
        or root.get("checkpoint_sha256") != WEIGHTS_SHA256
        or view not in VIEWS
        or root.get("conformers_per_molecule") != 1
    ):
        raise ValueError("Incompatible Atom-JEPA representation cache")
    folder = directory / view
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest != root["views"][view]:
        raise ValueError("Changed view manifest")
    for name, field in (("rows.csv", "rows_sha256"), ("embeddings.npy", "embeddings_sha256")):
        if sha256_file(folder / name) != manifest[field]:
            raise ValueError("Representation checksum mismatch")
    rows = pd.read_csv(folder / "rows.csv")
    indices = align_rows(rows, frame)
    matrix = np.load(folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (
        matrix.shape != (len(rows), VIEWS[view])
        or matrix.dtype != np.float32
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Invalid Atom-JEPA feature matrix")
    return np.array(matrix[indices]), root
