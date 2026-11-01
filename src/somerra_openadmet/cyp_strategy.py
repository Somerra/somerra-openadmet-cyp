"""Execute and score adaptive CYP strategies, preserving the legacy fit ledger."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .cyp_research import CYPS, leaderboard, read, run_trial, status
from .cyp_test import evaluate_test
from .download import sha256_file


def authorized_extension(path, root, cyp):
    """Read an explicit allocation extension without resetting frozen charges."""
    if path is None:
        return 0.0, None
    authorization = read(path)
    if Path(authorization["root"]).resolve() != Path(root).resolve():
        raise ValueError("Authorization belongs to another research root")
    deadline = datetime.fromisoformat(authorization["deadline_utc"])
    cap = read(Path(root) / "campaigns" / cyp / "campaign.json")["trial_seconds"]
    if (deadline - datetime.now(UTC)).total_seconds() < cap + 60:
        raise ValueError("Insufficient time before research deadline for fit and scoring")
    hours = float(authorization["additional_budget_hours"][cyp])
    if not 0 <= hours <= 24:
        raise ValueError("Invalid allocation extension")
    return hours, sha256_file(path)


def require_cuda():
    import torch

    subprocess.run(["nvidia-smi"], check=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is disabled")
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({"cuda": True, "free_bytes": free, "total_bytes": total}), flush=True)


def strategy_status(root):
    root = Path(root)
    combined = {"best": {}}
    for cyp in CYPS:
        path = root / "campaigns" / cyp / "strategy-registry.json"
        if path.exists():
            combined["best"].update(read(path)["best"])
    return {
        "selection_metric": "internal-test/ST-RAE",
        "score_interpretation": "adaptive development",
        **leaderboard(combined),
        "training_ledger_validation_only": status(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trial = sub.add_parser("trial")
    trial.add_argument("--root", type=Path, required=True)
    trial.add_argument("--cyp", choices=CYPS, required=True)
    trial.add_argument("--recipe", type=Path, required=True)
    trial.add_argument("--hypothesis", required=True)
    trial.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    trial.add_argument("--authorization", type=Path)
    score = sub.add_parser("score")
    score.add_argument("--trial", type=Path, required=True)
    score.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    view = sub.add_parser("status")
    view.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        result = strategy_status(args.root)
    else:
        require_cuda()
        if args.command == "trial":
            hours, digest = authorized_extension(args.authorization, args.root, args.cyp)
            hypothesis = args.hypothesis
            if digest:
                hypothesis += f"; allocation authorization sha256={digest}"
            record = run_trial(
                args.root,
                args.cyp,
                args.recipe,
                hypothesis,
                mode=args.wandb_mode,
                additional_budget_hours=hours,
            )
            if record["status"] != "complete":
                print(json.dumps(record), flush=True)
                raise SystemExit(1)
            path = Path(record["output"])
        else:
            path = args.trial
        result = evaluate_test(path, device="cuda", mode=args.wandb_mode)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
