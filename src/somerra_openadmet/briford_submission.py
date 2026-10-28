"""Authorized completion watcher: verify complete training, submit once, save score."""

import argparse
import datetime as dt
import fcntl
import json
import time
from pathlib import Path

from .briford import MEMBERS, check_contract, read, verify_complete, write
from .challenge_submission import live_preflight, submit
from .download import sha256_file


def follow_through(root):
    root = Path(root).resolve()
    state_path = root / "followthrough.json"
    with (root / "followthrough.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {
            "status": "waiting_for_training",
            "submission_authorized": True,
            "open_source_code": False,
            "model_report_link": "",
        }
        write(state_path, state)
        try:
            started = time.monotonic()
            while True:
                series = read(root / "series.json")
                if series["status"] == "predictions_prepared_not_submitted":
                    break
                if series["status"] in ("failed", "budget_exhausted"):
                    raise RuntimeError("Training stopped; no incomplete-ensemble submission")
                if time.monotonic() - started > 4.5 * 3600:
                    raise TimeoutError("Training completion deadline exceeded")
                time.sleep(30)
            manifest = check_contract(root)
            contract = read(root / "contract.json")
            if contract.get("submission_authorized") is not True:
                raise ValueError("Submission was not authorized")
            for member in MEMBERS:
                for k in range(5):
                    verify_complete(root, member, k)
            candidate = root / "predictions"
            export = read(candidate / "run_manifest.json")
            if export["contract_sha256"] != sha256_file(root / "contract.json"):
                raise ValueError("Prediction contract mismatch")
            receipt_path = candidate / "submission_receipt.json"
            state["status"] = "submitting_or_waiting_for_cooldown"
            write(state_path, state)
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt["status"] != "accepted":
                    raise ValueError(
                        "Prior submission state requires reconciliation; no automatic retry"
                    )
                if receipt["submission_sha256"] != export["submission_sha256"]:
                    raise ValueError("Existing receipt belongs to another prediction file")
            else:
                receipt = submit(candidate, manifest["official_raw"])
            state.update(status="accepted_waiting_for_score", receipt=str(receipt_path))
            write(state_path, state)
            accepted = dt.datetime.fromisoformat(receipt["submit_post_started_utc"])
            accepted = accepted.replace(second=0, microsecond=0)
            started = time.monotonic()
            while time.monotonic() - started < 2 * 3600:
                try:
                    portal = live_preflight()
                    write(candidate / "portal-after-submission.json", portal)
                    for row in portal["leaderboard_rows"]:
                        stamp = dt.datetime.strptime(row["Submitted"], "%Y-%m-%d %H:%M UTC")
                        stamp = stamp.replace(tzinfo=dt.UTC)
                        if stamp < accepted:
                            continue
                        link = str(row.get("Model Report Link", ""))
                        if "http" in link or "<a " in link or row.get("Open Code") != "No":
                            raise ValueError("Published metadata violates prediction-only policy")
                        state.update(
                            status="complete",
                            leaderboard_row=row,
                            briford_current_rows=portal.get("reference_rows", []),
                            briford_historical_score=0.4377997714479094,
                            historical_score_delta=float(row["MA-ST-RAE"]) - 0.4377997714479094,
                        )
                        write(state_path, state)
                        return state
                except ValueError:
                    raise
                except Exception as exc:
                    state["last_poll_error"] = str(exc)
                    write(state_path, state)
                time.sleep(60)
            state["status"] = "accepted_score_pending"
            write(state_path, state)
            return state
        except BaseException as exc:
            state.update(
                status="stopped_requires_attention", error_type=type(exc).__name__, error=str(exc)
            )
            write(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "requires": "20 verified networks and valid predictions",
                    "open_source_code": False,
                    "model_report_link": "",
                }
            )
        )
        return
    print(json.dumps(follow_through(args.output), indent=2))


if __name__ == "__main__":
    main()
