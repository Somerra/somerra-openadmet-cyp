"""Verify a pinned published submission; never treat it as a local research score."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ACTIVITY_ENDPOINTS
from .download import sha256_file
from .metrics import soft_threshold_rae

REFERENCE_COMMIT = "e70a2920d611684c9aaff07400d951e2e79dd5f6"
BUILD_SHA256 = "3c0e0f505cb98080ca0305d8bf8ecbdfef123b3bf75068018c43ffdd9f580368"
REFERENCE_URL = "https://github.com/Scigantic/open-admet-challenge-cyp"


def prediction_agreement(expected, rebuilt, *, tolerance=1e-4):
    """Require complete aligned predictions before applying the published tolerance."""
    columns = ["SMILES", "Molecule_Name", *ACTIVITY_ENDPOINTS]
    for frame in (expected, rebuilt):
        if list(frame.columns) != columns or frame.Molecule_Name.duplicated().any():
            raise ValueError("Unexpected prediction schema or duplicate molecule IDs")
        if frame.Molecule_Name.isna().any():
            raise ValueError("Missing molecule IDs")
        if not np.isfinite(frame[list(ACTIVITY_ENDPOINTS)].to_numpy(float)).all():
            raise ValueError("Non-finite predictions")
    if not expected.SMILES.equals(rebuilt.SMILES):
        raise ValueError("SMILES differ")
    if not expected.Molecule_Name.equals(rebuilt.Molecule_Name) or len(expected) == 0:
        raise ValueError("Molecule order or coverage differs")
    differences = np.abs(
        expected[list(ACTIVITY_ENDPOINTS)].to_numpy(float)
        - rebuilt[list(ACTIVITY_ENDPOINTS)].to_numpy(float)
    )
    maximum = float(differences.max())
    if maximum >= tolerance:
        raise ValueError(f"Prediction difference {maximum} exceeds tolerance {tolerance}")
    return {
        "rows": len(expected),
        "predictions": int(differences.size),
        "max_absolute_difference": maximum,
        "tolerance": tolerance,
        "per_endpoint_max_difference": dict(
            zip(ACTIVITY_ENDPOINTS, differences.max(0).tolist(), strict=True)
        ),
    }


def member_scores(source):
    """Rescore frozen member predictions using our shared metric, on upstream rows only."""
    scores = {}
    for endpoint in ACTIVITY_ENDPOINTS:
        short = endpoint.split("_")[0]
        truth = pd.read_csv(source / "data/oof_base" / f"r08_like_oof_{short}.csv")
        truth = truth.dropna(subset=["label", "conf_low", "conf_high", "r08_like"])
        if truth.Molecule_Name.duplicated().any():
            raise ValueError("Duplicate reference truth IDs")
        by_member = {}
        for path in sorted((source / "data/members").glob("*/oof.csv")):
            frame = pd.read_csv(path)
            column = endpoint.lower() + "_pred"
            if column not in frame:
                column = endpoint.lower()
            # The published rebuild averages repeated member predictions by molecule.
            pred = frame.groupby("Molecule_Name")[column].mean().reindex(truth.Molecule_Name)
            coverage = int(np.isfinite(pred).sum())
            by_member[path.parent.name] = {
                "observations": coverage,
                "ST-RAE": soft_threshold_rae(truth.label, pred, truth.conf_low, truth.conf_high)
                if coverage == len(truth)
                else None,
            }
        scores[endpoint] = {
            "observations": len(truth),
            "incumbent_ST-RAE": soft_threshold_rae(
                truth.label, truth.r08_like, truth.conf_low, truth.conf_high
            ),
            "members": by_member,
        }
    return scores


def reproduce(source: Path, output: Path, *, timeout_seconds=300):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the source checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != REFERENCE_COMMIT:
        raise ValueError(f"Expected reference commit {REFERENCE_COMMIT}, got {revision}")
    # Reject staged and unstaged edits to tracked inputs, while ignoring prior rebuild output.
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=source, check=True)
    if sha256_file(source / "scripts/build_r25.py") != BUILD_SHA256:
        raise ValueError("Reference build script changed")
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "snapshot"
    paths = ["scripts/build_r25.py", "data/MANIFEST.json"]
    for directory in ("data/members", "data/oof_base", "data/submissions"):
        paths += [
            str(path.relative_to(source)) for path in sorted((source / directory).rglob("*.csv"))
        ]
    input_hashes = {}
    for relative in paths:
        src, dst = source / relative, workspace / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        input_hashes[relative] = sha256_file(dst)
    (workspace / "notebooks").mkdir()
    record = {
        "status": "running",
        "score_kind": "frozen_submission_reproduction",
        "source_url": REFERENCE_URL,
        "source_commit": revision,
        "input_sha256": input_hashes,
        "versions": {p: version(p) for p in ("numpy", "pandas", "scipy")},
        "timeout_seconds": timeout_seconds,
        "gpu_hours": 0,
        "independent_local_score": None,
        "reported_leaderboard_MA-ST-RAE": 0.4436,
        "reported_score_verified_here": False,
    }
    manifest = output / "manifest.json"

    def save():
        manifest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    start = time.monotonic()
    try:
        with (output / "rebuild.log").open("x") as log:
            subprocess.run(
                [sys.executable, "scripts/build_r25.py"],
                cwd=workspace,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_seconds,
            )
        expected = workspace / "data/submissions/R25.csv"
        rebuilt = workspace / "notebooks/_build/R25_rebuilt.csv"
        agreement = prediction_agreement(pd.read_csv(expected), pd.read_csv(rebuilt))
        if agreement["rows"] != 750:
            raise ValueError("Expected 750 blind compounds")
        record.update(
            agreement=agreement,
            submitted_sha256=sha256_file(expected),
            rebuilt_sha256=sha256_file(rebuilt),
            byte_identical=sha256_file(expected) == sha256_file(rebuilt),
            upstream_fold_member_scores=member_scores(workspace),
            evaluation_note="Upstream scaffold rows; not our cluster folds or independent CV. "
            "R25 blind truth is unavailable. g26_tdi_armB has no published OOF predictions.",
        )
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        save()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    record = reproduce(args.source_dir, args.output_dir, timeout_seconds=args.timeout_seconds)
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in {"input_sha256", "upstream_fold_member_scores"}
            },
            indent=2,
        )
    )
    return record
