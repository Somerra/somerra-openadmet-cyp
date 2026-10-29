"""Versioned, identity-aligned representation caches and bounded run infrastructure.

Torch and encoder dependencies stay lazy so the ordinary CPU CLI remains usable.
Cache rows deliberately contain no labels or fold assignments.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import parent_smiles
from .constants import ACTIVITY_ENDPOINTS, TEST_FILE
from .download import sha256_file

CHEMELEON_SHA256 = "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"
CHEMELEON_FEATURIZER = "chemprop.MoleculeDataset default; original SMILES"
COMMON_KEYS = {
    "output_dir",
    "device",
    "precision",
    "memory_fraction",
    "free_memory_gib",
    "max_gpu_hours",
    "campaign_ledger",
    "threads",
    "seed",
}


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_config(path: Path, allowed: set[str], required: set[str]) -> dict:
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    if set(config) - allowed - COMMON_KEYS:
        raise ValueError(f"Unknown configuration keys: {set(config) - allowed - COMMON_KEYS}")
    if required - set(config):
        raise ValueError(f"Missing configuration keys: {required - set(config)}")
    return config


def provenance() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "rdkit", "torch", "chemprop"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "versions": versions,
        "source_sha256": {
            p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def identities(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[["Molecule_Name", "SMILES"]].copy().reset_index(drop=True)
    if rows.empty or rows.isna().any(axis=None):
        raise ValueError("Missing molecular identities or empty input")
    if not rows.Molecule_Name.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Molecule IDs must be nonempty strings")
    if rows.Molecule_Name.duplicated().any():
        raise ValueError("Duplicate molecule IDs")
    if not rows.SMILES.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Missing SMILES")
    rows["smiles_sha256"] = rows.SMILES.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    rows["parent"] = rows.SMILES.map(parent_smiles)
    rows["valid"] = True
    return rows


def prepared_data(directory: Path):
    evidence = json.loads((directory / "manifest.json").read_text())
    path = directory / "training.csv"
    if sha256_file(path) != evidence["training_sha256"]:
        raise ValueError("Prepared training checksum mismatch")
    frame = pd.read_csv(path)
    rows = identities(frame)
    if not np.array_equal(rows.parent, frame.parent):
        raise ValueError("Prepared parent identities changed")
    if frame[["cluster", "fold"]].isna().any(axis=None):
        raise ValueError("Missing split assignments")
    if frame.groupby("cluster").fold.nunique().max() != 1:
        raise ValueError("A molecular cluster crosses folds")
    if frame.groupby("parent").cluster.nunique().max() != 1:
        raise ValueError("A molecular parent crosses clusters")
    if set(frame.fold.unique()) != set(range(evidence["n_folds"])):
        raise ValueError("Unexpected fold IDs")
    for endpoint in ACTIVITY_ENDPOINTS:
        observed = frame[endpoint].notna()
        values = frame.loc[observed, [endpoint, endpoint + "_conf_low", endpoint + "_conf_high"]]
        if values.empty or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Invalid observed labels/bounds: {endpoint}")
        if (values.iloc[:, 1] > values.iloc[:, 2]).any():
            raise ValueError("Reversed confidence bounds")
    return frame, evidence


def foundation(path: Path):
    from .neural import encoder

    if sha256_file(path) != CHEMELEON_SHA256:
        raise ValueError("Expected the pinned general CheMeleon foundation checkpoint")
    model = encoder(path)
    if model.output_dim != 2048:
        raise ValueError("Unexpected CheMeleon embedding dimension")
    return model


def batch_indices(indices, size: int, *, merge_singleton=False):
    if type(size) is not int or size < 2:
        raise ValueError("batch_size must be an integer >= 2")
    indices = np.asarray(indices, dtype=int)
    batches = [indices[i : i + size] for i in range(0, len(indices), size)]
    if merge_singleton and batches and len(batches[-1]) == 1:
        if len(batches) == 1:
            raise ValueError("BatchNorm training requires at least two rows")
        batches[-2:] = [np.concatenate(batches[-2:])]
    return batches


def graph_dataset(smiles):
    from chemprop import data

    dataset = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles])
    dataset.cache = True
    return dataset


def graph_batch(dataset, indices, device):
    from chemprop import data

    graph = data.collate_batch([dataset[int(i)] for i in indices]).bmg
    # Chemprop's .to() does not expose non_blocking; pin/transfer the tensors here.
    for name in ("V", "E", "edge_index", "rev_edge_index", "batch"):
        tensor = getattr(graph, name)
        if device == "cuda":
            tensor = tensor.pin_memory()
        setattr(graph, name, tensor.to(device, non_blocking=device == "cuda"))
    return graph


class Run:
    """Write incomplete evidence first, reserve before GPU work, finalize on every exit."""

    def __init__(self, config):
        import torch

        self.config = config
        self.output = Path(config["output_dir"])
        requested = config.get("device", "auto")
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.hours = float(config.get("max_gpu_hours", 0))
        if not np.isfinite(self.hours) or not 0 < self.hours <= 12:
            raise ValueError("Specify max_gpu_hours in (0, 12]; this is also the CPU wall-time cap")
        threads = config.get("threads", 8)
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)
        # Float32 evaluation/replay must not silently use TF32 matmul rounding.
        torch.set_float32_matmul_precision("highest")
        self.record = {
            "schema_version": 1,
            "config": config,
            "complete": False,
            "device": self.device,
            "gpu_hours": 0,
            "purpose": "screening_only",
            "open_source_code": False,
            "model_report_link": "",
            **provenance(),
        }
        self.allocation = None

    def __enter__(self):
        import torch

        from .campaign_budget import WorkerBudget
        from .representation_budget import ledger_path

        self.output.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        self.deadline = self.start + self.hours * 3600
        self.save()
        try:
            if self.device == "cuda":
                shared_ledger = ledger_path(self.config)
                if os.environ.get("SOMERRA_REPRESENTATION_WORKER") != str(self.output.resolve()):
                    raise ValueError("GPU runs require the shared campaign supervisor")
                allocation = WorkerBudget(shared_ledger, self.output, self.hours)
                allocation.reserve()
                self.allocation = allocation
                self.record["campaign_ledger"] = str(shared_ledger)
                self.record["accounting_scope"] = "worker elapsed; coordinator settles full process"
                free, total = torch.cuda.mem_get_info()
                fraction = float(self.config.get("memory_fraction", 0.8))
                headroom = float(self.config.get("free_memory_gib", 10)) * 2**30
                if not 0 < fraction <= 0.8 or not np.isfinite(headroom) or headroom < 10 * 2**30:
                    raise ValueError("Use memory_fraction <= 0.8 and free_memory_gib >= 10")
                allowance = min(total * fraction, free - headroom)
                if allowance <= 0:
                    raise RuntimeError("Insufficient free GPU memory after reserved headroom")
                torch.cuda.set_per_process_memory_fraction(allowance / total)
                torch.cuda.reset_peak_memory_stats()
                self.record["hardware"] = {
                    "name": torch.cuda.get_device_name(),
                    "total_bytes": total,
                    "initial_free_bytes": free,
                    "allocator_limit_bytes": int(allowance),
                    "reserved_headroom_bytes": int(headroom),
                }
            self.save()
            return self
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def check(self, projected_seconds=0):
        if time.monotonic() + projected_seconds >= self.deadline:
            raise TimeoutError("Insufficient remaining run allocation")

    def sync(self):
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()

    def save(self):
        atomic_json(self.output / "run_manifest.json", self.record)

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.record.update(complete=False, error=f"{kind.__name__}: {error}")
        self.record["wall_seconds"] = time.monotonic() - self.start
        if self.allocation is not None:
            self.record["gpu_hours"] = self.allocation.finish()
            import torch

            self.record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            self.record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        self.save()


@contextmanager
def autocast(device: str, precision: str):
    import torch

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=precision == "bf16"):
        yield


def requested_precision(config, device):
    import torch

    value = config.get("precision", "float32")
    if value not in {"float32", "bf16"}:
        raise ValueError("precision must be float32 or bf16")
    if value == "bf16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        return "float32"
    return value


def write_cache(output, rows, matrix, metadata):
    matrix = np.asarray(matrix)
    if matrix.dtype != np.float32 or len(matrix) != len(rows) or not np.isfinite(matrix).all():
        raise ValueError("Invalid representation array")
    rows.to_csv(output / "rows.csv.partial", index=False)
    with (output / "embeddings.npy.partial").open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    os.replace(output / "rows.csv.partial", output / "rows.csv")
    os.replace(output / "embeddings.npy.partial", output / "embeddings.npy")
    manifest = {
        **metadata,
        "schema_version": 1,
        "complete": True,
        "shape": list(matrix.shape),
        "dtype": "float32",
        "rows_sha256": sha256_file(output / "rows.csv"),
        "embeddings_sha256": sha256_file(output / "embeddings.npy"),
    }
    # The manifest is the commit marker. Readers reject any interrupted/incomplete cache.
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_cache(directory: Path, frame: pd.DataFrame, *, checkpoint_sha256=CHEMELEON_SHA256):
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {
        "schema_version": 1,
        "complete": True,
        "encoder": "chemeleon",
        "representation": "mean",
        "featurizer": CHEMELEON_FEATURIZER,
        "checkpoint_sha256": checkpoint_sha256,
        "dtype": "float32",
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Incompatible representation cache provenance")
    for filename, field in (("rows.csv", "rows_sha256"), ("embeddings.npy", "embeddings_sha256")):
        if sha256_file(directory / filename) != manifest[field]:
            raise ValueError("Representation cache checksum mismatch")
    rows = pd.read_csv(directory / "rows.csv")
    recomputed = identities(rows)
    if not rows[recomputed.columns].equals(recomputed):
        raise ValueError("Invalid cached row identities")
    matrix = np.load(directory / "embeddings.npy", allow_pickle=False, mmap_mode="r")
    if (
        list(matrix.shape) != manifest["shape"]
        or matrix.dtype != np.float32
        or matrix.shape != (len(rows), 2048)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Invalid cached representation dimensions/values")
    query = identities(frame)
    indexed = rows.set_index("Molecule_Name")
    if not set(query.Molecule_Name).issubset(indexed.index):
        raise ValueError("Cache missing requested molecule IDs")
    aligned = indexed.loc[query.Molecule_Name]
    if not np.array_equal(aligned.smiles_sha256, query.smiles_sha256):
        raise ValueError("Cache SMILES do not match requested identities")
    positions = pd.Series(np.arange(len(rows)), index=rows.Molecule_Name).loc[query.Molecule_Name]
    return np.asarray(matrix[positions.to_numpy()]), manifest


def profile_extraction(model, aggregate, dataset, rows, precision, run):
    """Profile inference separately; no labels or training-selected batch size are used."""
    import torch

    order = np.argsort(-rows.SMILES.str.len().to_numpy())
    records = []
    for size in sorted({min(n, len(rows)) for n in (64, 128, 256, 512, 1024)}):
        graph = encoded = None
        result = {"batch_size": size, "status": "oom"}
        try:
            with torch.inference_mode():
                for iteration in range(7):
                    run.check()
                    if iteration == 2:
                        run.sync()
                        started = time.monotonic()
                    graph = graph_batch(dataset, order[:size], run.device)
                    with autocast(run.device, precision):
                        encoded = aggregate(model(graph), graph.batch)
                run.sync()
                seconds = time.monotonic() - started
                if not torch.isfinite(encoded).all():
                    raise ValueError("Non-finite extraction profile")
            result.update(status="ok", molecules_per_second=5 * size / seconds)
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            del graph, encoded
            if run.device == "cuda":
                torch.cuda.empty_cache()
        records.append(result)
        atomic_json(run.output / "extraction-profile.json", records)
    good = [r for r in records if r["status"] == "ok"]
    if not good:
        raise RuntimeError("No extraction batch fits the memory allowance")
    best = max(r["molecules_per_second"] for r in good)
    return min(r["batch_size"] for r in good if r["molecules_per_second"] >= best * 0.95)


def cache_representations(config_path: Path):
    from .representation_budget import maybe_supervise

    supervised = maybe_supervise("cache", config_path)
    if supervised is not None:
        return supervised

    if json.loads(Path(config_path).read_text()).get("encoder") == "mol_jepa":
        from .mol_jepa import cache_representations as extract

        return extract(config_path)
    import torch
    from chemprop import nn

    config = read_config(
        config_path,
        {"prepared_dir", "blind_csv", "checkpoint", "batch_size"},
        {"prepared_dir", "blind_csv", "checkpoint", "output_dir", "max_gpu_hours"},
    )
    train, evidence = prepared_data(Path(config["prepared_dir"]))
    blind_path = Path(config["blind_csv"])
    if sha256_file(blind_path) != evidence["input_sha256"][TEST_FILE]:
        raise ValueError("Blind input differs from prepared data manifest")
    blind = pd.read_csv(blind_path)
    rows = identities(pd.concat([train, blind], ignore_index=True))
    if set(rows.parent.iloc[: len(train)]) & set(rows.parent.iloc[len(train) :]):
        raise ValueError("Training parents overlap blind parents")
    model = foundation(Path(config["checkpoint"]))
    dataset = graph_dataset(rows.SMILES)
    batch_size = config.get("batch_size", "auto")
    if batch_size != "auto":
        batch_indices(np.arange(len(rows)), batch_size)
    with Run(config) as run:
        model = model.to(run.device).eval()
        aggregate = nn.MeanAggregation()
        precision = requested_precision(config, run.device)
        first = graph_batch(dataset, np.arange(min(64, len(rows))), run.device)
        with torch.inference_mode():
            baseline = aggregate(model(first), first.batch).float()
            with autocast(run.device, precision):
                probe = aggregate(model(first), first.batch).float()
            error = float((baseline - probe).abs().max())
            if not torch.isfinite(probe).all() or not torch.allclose(
                baseline, probe, atol=0.01, rtol=0.01
            ):
                precision = "float32"
        del first, baseline, probe
        run.record.update(precision=precision, precision_probe_max_abs=error)
        if batch_size == "auto":
            batch_size = profile_extraction(model, aggregate, dataset, rows, precision, run)
        batches = batch_indices(np.arange(len(rows)), batch_size)
        output, timings = [], []
        with torch.inference_mode():
            for index, indices in enumerate(batches):
                run.check()
                run.sync()
                started = time.monotonic()
                graph = graph_batch(dataset, indices, run.device)
                with autocast(run.device, precision):
                    encoded = aggregate(model(graph), graph.batch).float()
                output.append(encoded.cpu().numpy())
                run.sync()
                timings.append(time.monotonic() - started)
                if index == 0:
                    run.check(timings[0] * (len(batches) - 1) * 1.2)
        metadata = {
            "encoder": "chemeleon",
            "representation": "mean",
            "checkpoint_sha256": CHEMELEON_SHA256,
            "precision": precision,
            "featurizer": CHEMELEON_FEATURIZER,
            "input_sha256": {
                "training": evidence["training_sha256"],
                "blind": sha256_file(blind_path),
            },
            "encoder_parameters": {k: v for k, v in model.hparams.items() if k != "cls"},
            **provenance(),
        }
        result = write_cache(run.output, rows, np.concatenate(output), metadata)
        run.record.update(
            complete=True,
            rows=len(rows),
            batch_size=batch_size,
            molecules_per_second=len(rows) / sum(timings),
            cache_manifest_sha256=sha256_file(run.output / "manifest.json"),
        )
        return result
