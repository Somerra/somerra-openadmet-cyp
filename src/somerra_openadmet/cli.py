"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import audit_data, load_test
from .download import download_activity_data, verify_activity_data
from .ensemble import run_activity_ensemble
from .metrics import score_activity, score_tdi
from .models import CLASSIFICATION_MODELS, REGRESSION_MODELS
from .pipeline import BaselineConfig, run_baseline
from .validation import validate_submission


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openadmet-cyp")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download pinned activity-track data")
    download.add_argument("--output", type=Path, default=Path("data/raw"))
    download.add_argument("--revision", default=None)

    audit = commands.add_parser("audit", help="Validate and summarize local challenge data")
    audit.add_argument("--data-dir", type=Path, default=Path("data/raw"))

    verify = commands.add_parser("verify-data", help="Verify downloaded files against manifest")
    verify.add_argument("--data-dir", type=Path, default=Path("data/raw"))

    baseline = commands.add_parser("baseline", help="Run CV, fit full models, and build a CSV")
    baseline.add_argument("track", choices=("activity", "tdi"))
    baseline.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--model", default="lightgbm")
    baseline.add_argument("--n-splits", type=int, default=5)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--radius", type=int, default=2)
    baseline.add_argument("--n-bits", type=int, default=2048)
    baseline.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Maximum model-training threads (recorded in the run manifest)",
    )

    ensemble = commands.add_parser(
        "ensemble", help="Cross-fit metric-aligned weights over completed baseline runs"
    )
    ensemble.add_argument("track", choices=("activity",))
    ensemble.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        action="append",
        required=True,
        help="Two or more baseline run directories; the flag may be repeated",
    )
    ensemble.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ensemble.add_argument("--output-dir", type=Path, required=True)
    ensemble.add_argument("--seed", type=int, default=42)
    ensemble.add_argument("--search-samples", type=int, default=4096)

    validate = commands.add_parser("validate", help="Validate a submission against test IDs")
    validate.add_argument("track", choices=("activity", "tdi"))
    validate.add_argument("submission", type=Path)
    validate.add_argument("--data-dir", type=Path, default=Path("data/raw"))

    score = commands.add_parser("score", help="Score labeled holdout predictions locally")
    score.add_argument("track", choices=("activity", "tdi"))
    score.add_argument("truth", type=Path)
    score.add_argument("predictions", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        kwargs = {} if args.revision is None else {"revision": args.revision}
        print(download_activity_data(args.output, **kwargs))
        return
    if args.command == "audit":
        _json(audit_data(args.data_dir).to_dict())
        return
    if args.command == "verify-data":
        report = verify_activity_data(args.data_dir)
        _json(report.to_dict())
        if not report.valid:
            raise SystemExit(1)
        return
    if args.command == "baseline":
        allowed = REGRESSION_MODELS if args.track == "activity" else CLASSIFICATION_MODELS
        if args.model not in allowed:
            raise SystemExit(f"--model must be one of: {', '.join(allowed)}")
        config = BaselineConfig(
            track=args.track,
            model=args.model,
            n_splits=args.n_splits,
            seed=args.seed,
            radius=args.radius,
            n_bits=args.n_bits,
            n_jobs=args.n_jobs,
        )
        _json(run_baseline(config, args.data_dir, args.output_dir))
        return
    if args.command == "ensemble":
        run_dirs = [run_dir for group in args.run_dir for run_dir in group]
        _json(
            run_activity_ensemble(
                run_dirs,
                args.data_dir,
                args.output_dir,
                seed=args.seed,
                search_samples=args.search_samples,
            )
        )
        return
    if args.command == "validate":
        report = validate_submission(args.submission, load_test(args.data_dir), args.track)
        _json(report.to_dict())
        if not report.valid:
            raise SystemExit(1)
        return
    if args.command == "score":
        truth = pd.read_csv(args.truth)
        predictions = pd.read_csv(args.predictions)
        result = (
            score_activity(truth, predictions)
            if args.track == "activity"
            else score_tdi(truth, predictions)
        )
        _json(result)


if __name__ == "__main__":
    main()
