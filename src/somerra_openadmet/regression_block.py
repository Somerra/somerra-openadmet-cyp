"""Bounded proposal schedule around the frozen single-candidate autoresearch evaluator."""

import argparse
import csv
import html
import itertools
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .campaign_budget import LIMIT_HOURS, CampaignLedger
from .model_recipe import resolve_model
from .regression_task import ROOT, atomic_json, checked_run, read_registry, read_task


def proposals(base):
    axes = {
        "scored_endpoint_weight": [2.0, 1.5, 4.0, 0.5],
        "max_lr": [0.0005, 0.002, 0.0003, 0.003],
        "dropout": [0.1, 0.3, 0.0, 0.4],
        "depth": [3, 5, 2, 6],
        "hidden_dim": [1024, 3072, 512],
        "ffn_hidden_dim": [512, 2048],
        "ffn_layers": [1, 3],
        "auxiliary_weight": [0.75, 1.25],
    }
    return [
        {
            "name": f"{key}-{value}",
            "stage": "single_axis",
            "changes": {key: value},
            "model": resolve_model({**base, key: value}),
        }
        for key, values in axes.items()
        for value in values
    ]


def refinements(base, completed, reference_validation):
    """Choose follow-ups by validation only, never by internal-test point/CI."""
    eligible = sorted(
        [r for r in completed if r.get("validation") is not None and r["stage"] == "single_axis"],
        key=lambda r: (r["validation"], r["name"]),
    )
    winners = {}
    for row in eligible:
        key = next(iter(row["changes"]))
        if row["validation"] < reference_validation and key not in winners:
            winners[key] = row["changes"][key]
        if len(winners) == 4:
            break
    results = []
    for size in range(2, len(winners) + 1):
        for keys in itertools.combinations(winners, size):
            changes = {key: winners[key] for key in keys}
            results.append(
                {
                    "name": "combine-" + "-".join(keys),
                    "stage": "combination",
                    "changes": changes,
                    "model": resolve_model({**base, **changes}),
                }
            )
    best = (
        eligible[0]["model"]
        if eligible and eligible[0]["validation"] < reference_validation
        else base
    )
    for key, values in {
        "scored_endpoint_weight": [0.75, 3.0],
        "max_lr": [0.00075, 0.0015],
        "dropout": [0.05, 0.15, 0.25],
        "hidden_dim": [1536],
        "ffn_hidden_dim": [768],
        "auxiliary_weight": [1.5],
    }.items():
        results.extend(
            {
                "name": f"refine-{key}-{v}",
                "stage": "refinement",
                "changes": {key: v},
                "model": resolve_model({**best, key: v}),
            }
            for v in values
        )
    for weight in [2.0, 4.0]:
        changes = {"loss_kind": "interval", "scored_endpoint_weight": weight}
        results.append(
            {
                "name": f"interval-scored-{weight}",
                "stage": "interaction",
                "changes": changes,
                "model": resolve_model({**base, **changes}),
            }
        )
    return results


def allowance(remaining_seconds, durations):
    expected = max(600, statistics.median(durations[-5:]) * 1.5) if durations else 600
    if remaining_seconds < expected + 120:
        return None
    return min(0.4, (remaining_seconds - 120) / 3600)


def report(output, state, task_path, task):
    selected = read_registry(task_path, task)["incumbent"]
    atomic_json(output / "recommendation.json", selected)
    atomic_json(output / "best-model.json", selected["model_config"])
    fields = ["name", "stage", "status", "validation", "score", "best_epoch", "decision", "seconds"]
    with (output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state["trials"])
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in fields) + "</tr>"
        for row in state["trials"]
    )
    page = (
        '<!doctype html><html><head><meta name="viewport" '
        'content="width=device-width,initial-scale=1">'
        "<title>Regression experiment block</title><style>body{font:16px system-ui;margin:20px;"
        "max-width:1100px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}"
        ".table{overflow-x:auto}code{overflow-wrap:anywhere}</style></head><body>"
        f"<h1>Regression experiment block</h1><p>Status: {html.escape(state['status'])}</p>"
        f"<p>Recommended single model internal-test MA-ST-RAE: <b>{selected['score']:.6f}</b>.</p>"
        f"<p>Reference: {state['reference_score']:.6f}. Lower is better.</p>"
        "<p>Seed 42; frozen 60/20/20 molecular split; 100 epochs. Validation selects checkpoints "
        "and follow-up proposals. Internal test is an adaptive development objective, not the "
        "official challenge score. Promotions require at least 3% gain, no endpoint regression "
        "over 0.02, and a paired 95% upper delta below zero. "
        "No challenge submission or final fit.</p>"
        f"<p>Selected run: <code>{html.escape(selected['run'])}</code></p>"
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{k}</th>" for k in fields)
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table></div></body></html>"
    )
    (output / "report.html").write_text(page)


def run(task_path, output, hours=4.0, max_trials=40):
    if not 0 < hours <= 4 or not 1 <= max_trials <= 40:
        raise ValueError("Block must be at most four hours and 40 trials")
    task_path, output = Path(task_path).resolve(), Path(output).resolve()
    task = read_task(task_path, check_runtime=True)
    incumbent = read_registry(task_path, task)["incumbent"]
    base = resolve_model(incumbent["model_config"])
    reference = checked_run(incumbent["run"])
    ledger = CampaignLedger(task["campaign_ledger"])
    starting = ledger.snapshot()
    output.mkdir(parents=True, exist_ok=False)
    start, start_wall = time.monotonic(), time.time()
    state = {
        "status": "running",
        "task": str(task_path),
        "hours": hours,
        "started_at_unix": start_wall,
        "deadline_unix": start_wall + hours * 3600,
        "reference_score": incumbent["score"],
        "reference_run": incumbent["run"],
        "reference_validation": reference["best_score"],
        "starting_budget": starting,
        "max_trials": max_trials,
        "trials": [],
    }
    queue = proposals(base)
    atomic_json(
        output / "plan.json",
        {
            "initial_trials": queue,
            "follow_up_selection": "validation only",
            "max_trials": max_trials,
            "hours": hours,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )
    seen, durations, expanded, failures = {json.dumps(base, sort_keys=True)}, [], False, 0
    try:
        while len(state["trials"]) < max_trials:
            if not queue:
                if expanded:
                    state["status"] = "completed_schedule"
                    break
                queue = refinements(base, state["trials"], reference["best_score"])
                expanded = True
                atomic_json(output / "follow-up-plan.json", queue)
            if not queue:
                continue
            proposal = queue.pop(0)
            key = json.dumps(proposal["model"], sort_keys=True)
            if key in seen:
                continue
            remaining = hours * 3600 - (time.monotonic() - start)
            available = LIMIT_HOURS - ledger.snapshot()["committed_hours"]
            allocation = allowance(min(remaining, available * 3600), durations)
            if allocation is None:
                state["status"] = "completed_budget_boundary"
                break
            if shutil.disk_usage(output).free < 2 * 1024**3:
                state["status"] = "stopped_low_disk"
                break
            seen.add(key)
            index = len(state["trials"]) + 1
            row = {
                **proposal,
                "status": "running",
                "output": str(output / f"trial-{index:02d}"),
                "allowance_hours": allocation,
                "started_at_unix": time.time(),
            }
            state["trials"].append(row)
            atomic_json(output / "block.json", state)
            config = output / f"recipe-{index:02d}.json"
            atomic_json(config, row["model"])
            command = [
                sys.executable,
                str(ROOT / "autoresearch.py"),
                "run",
                "--task",
                str(task_path),
                "--model-config",
                str(config),
                "--output-dir",
                row["output"],
                "--hours",
                str(allocation),
                "--hypothesis",
                f"{row['stage']}: {row['name']}. Frozen split, seed and epoch policy; "
                "follow-up proposals selected using validation only.",
            ]
            trial_start = time.monotonic()
            with (output / f"trial-{index:02d}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                )
            row.update(
                seconds=time.monotonic() - trial_start,
                returncode=result.returncode,
                status="complete" if result.returncode == 0 else "failed",
            )
            durations.append(row["seconds"])
            trial_file = Path(row["output"]) / "trial.json"
            if trial_file.exists():
                trial = json.loads(trial_file.read_text())
                row.update({k: trial.get(k) for k in ["decision", "score", "compared_to", "error"]})
                if trial.get("status") == "complete":
                    record = checked_run(trial["run"])
                    row.update(validation=record["best_score"], best_epoch=record["best_epoch"])
            failures = failures + 1 if result.returncode else 0
            atomic_json(output / "block.json", state)
            report(output, state, task_path, task)
            if failures >= 2:
                state["status"] = "stopped_repeated_failures"
                break
        else:
            state["status"] = "completed_trial_limit"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state.update(
            finished_at_unix=time.time(),
            elapsed_seconds=time.monotonic() - start,
            ending_budget=ledger.snapshot(),
        )
        atomic_json(output / "block.json", state)
        report(output, state, task_path, task)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=4.0)
    args = parser.parse_args()
    run(args.task, args.output, args.hours)


if __name__ == "__main__":
    main()
