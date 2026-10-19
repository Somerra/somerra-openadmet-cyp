"""Editable regression experiment recipe; the evaluator lives in src/ and stays fixed.

See program.md before running. Each invocation runs ONE experiment, never a loop.
"""

import argparse
import json
from pathlib import Path

from somerra_openadmet.configuration import resolve_config

# Model research changes model knobs; data research changes only data_sources.
BACKEND = "cpu"  # "cpu" (nested evaluation) or "neural" (screening only)
MODEL_RECIPE = Path("configs/models/incumbent.json")
MODEL_OVERRIDES = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if BACKEND not in {"cpu", "neural"}:
        raise ValueError("BACKEND must be cpu or neural")
    model = {**json.loads(MODEL_RECIPE.read_text()), **MODEL_OVERRIDES}
    if "selected_folds" in model:
        raise ValueError("Research trials require all folds; recovery uses the separate CLI")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    # Sibling snapshots keep all trial outputs directly under runs/, sharing the
    # existing neural budget ledger. No per-trial recipes are committed.
    inputs = args.output_dir.with_name(args.output_dir.name + "-inputs")
    inputs.mkdir(parents=True, exist_ok=False)
    recipe = inputs / "model.json"
    data = inputs / "data.json"
    recipe.write_text(json.dumps(model, indent=2) + "\n")
    data.write_text(args.data_config.read_text())
    (inputs / "train.py").write_text(Path(__file__).read_text())
    resolve_config(model_config=recipe, data_config=data, output_dir=args.output_dir)
    if BACKEND == "cpu":
        from somerra_openadmet.campaign import evaluate as run
    else:
        from somerra_openadmet.neural import screen as run
    result = run(model_config=recipe, data_config=data, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
