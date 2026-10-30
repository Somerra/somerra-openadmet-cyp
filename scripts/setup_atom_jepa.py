"""Fetch pinned Atom-JEPA assets, verify bytes, and create local provenance.

No GPU work or remote Python execution. Install the verified source separately.
Existing directories are verified in place, never overwritten or checked out anew.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PIN = Path(__file__).resolve().parents[1] / "configs/atom-jepa-provenance.json"


def verify_files(root, expected):
    for name, digest in expected.items():
        path = root / name
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {path}")


def verify_source(source, pin):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != pin["source_revision"]:
        raise ValueError("Source revision differs from the reviewed pin")
    verify_files(source / "atom_jepa", pin["source_files"])


def setup(source, model, *, verify_only=False):
    pin = json.loads(PIN.read_text())
    if not source.exists():
        if verify_only:
            raise FileNotFoundError(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        staging = source.with_name(source.name + ".partial")
        if staging.exists():
            raise FileExistsError(f"Investigate the unfinished download: {staging}")
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/khelverskovp/atom-jepa.git",
                str(staging),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(staging), "checkout", "--detach", pin["source_revision"]],
            check=True,
        )
        verify_source(staging, pin)
        staging.rename(source)
    verify_source(source, pin)
    if not model.exists():
        if verify_only:
            raise FileNotFoundError(model)
        from huggingface_hub import hf_hub_download

        staging = model.with_name(model.name + ".partial")
        staging.mkdir(parents=True, exist_ok=False)
        for name in pin["files"]:
            cached = hf_hub_download("atom-jepa/atom-jepa", name, revision=pin["model_revision"])
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
        verify_files(staging, pin["files"])
        (staging / "provenance.json").write_text(json.dumps(pin, indent=2) + "\n")
        staging.rename(model)
    verify_files(model, pin["files"])
    if json.loads((model / "provenance.json").read_text()) != pin:
        raise ValueError("Existing provenance differs from the reviewed pin")
    return {
        "source_revision": pin["source_revision"],
        "model_revision": pin["model_revision"],
        "verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("runs/atom-jepa-source"))
    parser.add_argument("--model-dir", type=Path, default=Path("runs/atom-jepa-model"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(setup(args.source_dir, args.model_dir, verify_only=args.verify_only), indent=2)
    )
