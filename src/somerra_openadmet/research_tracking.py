"""Private, scalar-only W&B tracking with an authoritative local event journal."""

from __future__ import annotations

import json
from pathlib import Path


def ensure_private_project(wandb, entity, project):
    from wandb_gql import gql

    api = wandb.Api(timeout=20)
    query = gql("""query($entity: String!, $project: String!) {
        project(name: $project, entityName: $entity) { name access }
    }""")
    variables = {"entity": entity, "project": project}
    result = api.client.execute(query, variable_values=variables)["project"]
    if result is None:
        api.client.execute(
            gql("""mutation($entity: String!, $project: String!) {
            upsertModel(input: {entityName: $entity, name: $project, access: "PRIVATE"}) {
                model { name access }
            }
        }"""),
            variable_values=variables,
        )
        result = api.client.execute(query, variable_values=variables)["project"]
    if not result or result["access"] != "PRIVATE":
        raise ValueError("W&B destination must be private before telemetry is sent")


class Tracker:
    def __init__(self, output, tracking, *, run_id, group, cyp, config, mode, smoke=False):
        self.output, self.run = Path(output), None
        self.path = self.output / "events.jsonl"
        self.error = None
        if mode == "disabled":
            return
        try:
            import wandb

            if mode == "online":
                ensure_private_project(wandb, tracking["entity"], tracking["project"])
            self.run = wandb.init(
                entity=tracking["entity"],
                project=tracking["project"],
                id=run_id,
                name=run_id,
                group=group,
                job_type="smoke" if smoke else config.get("workflow", "research"),
                tags=(
                    [cyp, "internal-test", "report-only"]
                    if config.get("workflow") == "internal-test"
                    else [cyp, "development-fold", "smoke" if smoke else "candidate"]
                ),
                config=config,
                dir=str(self.output),
                mode=mode,
                save_code=False,
                resume="allow" if mode == "online" else None,
                settings=wandb.Settings(
                    disable_code=True,
                    disable_git=True,
                    console="off",
                    x_disable_meta=True,
                    init_timeout=30,
                ),
            )
            self.run.define_metric("epoch")
            for prefix in ["train", "val", "loss", "diagnostic_mae", "optimizer", "runtime"]:
                self.run.define_metric(f"{prefix}/*", step_metric="epoch")
            self.run.define_metric("val/ST-RAE", summary="min")
            (self.output / "tracking.json").write_text(
                json.dumps(
                    {
                        "id": self.run.id,
                        "url": self.run.url,
                        "mode": mode,
                        "entity": tracking["entity"],
                        "project": tracking["project"],
                    },
                    indent=2,
                )
                + "\n"
            )
        except Exception as exc:
            # Never initialize an online run if privacy could not be verified.
            # Tracking availability must not decide whether a scientific trial succeeds.
            if self.run is not None:
                try:
                    self.run.finish(exit_code=1)
                except Exception:
                    pass
            self.run = None
            self.error = type(exc).__name__
            (self.output / "tracking-error.json").write_text(
                json.dumps(
                    {
                        "error_type": self.error,
                        "phase": "initialization",
                        "requested_mode": mode,
                        "effective_mode": "disabled",
                        "local_events_preserved": True,
                    }
                )
            )

    def log(self, values):
        with self.path.open("a") as handle:
            handle.write(json.dumps(values, allow_nan=False) + "\n")
        if self.run is not None:
            try:
                self.run.log({k: v for k, v in values.items() if v is not None})
            except Exception as exc:
                self.error = type(exc).__name__
                (self.output / "tracking-error.json").write_text(
                    json.dumps(
                        {
                            "error_type": self.error,
                            "local_events_preserved": True,
                        }
                    )
                )

    def finish(self, summary, failed=False):
        if self.run is not None:
            try:
                self.run.summary.update(summary)
                self.run.finish(exit_code=int(failed))
            except Exception as exc:
                (self.output / "tracking-error.json").write_text(
                    json.dumps(
                        {
                            "error_type": type(exc).__name__,
                            "local_events_preserved": True,
                        }
                    )
                )


def log_campaign(root, manifest, registry, mode, *, all_best=None):
    from .cyp_research import leaderboard

    # Smoke tests are always isolated from scientific score comparisons.
    if not registry["best"]:
        return
    output = Path(root) / "campaign-tracking"
    output.mkdir(exist_ok=True)
    # Read-only cross-CYP context; selection and spending remain independent.
    board = leaderboard({"best": all_best}) if all_best is not None else leaderboard(registry)
    values = {
        "trial_index": len(registry["trials"]),
        "campaign/completed_objectives": board["completed_objectives"],
    }
    if board["MA-ST-RAE"] is not None:
        values["campaign/MA-ST-RAE"] = board["MA-ST-RAE"]
    for cyp, entry in board["best"].items():
        if entry:
            values.update({f"best/{cyp}/{k}": v for k, v in entry["metrics"].items()})
    try:
        tracker = Tracker(
            output,
            manifest["config"]["wandb"],
            run_id="campaign-" + manifest["campaign_id"],
            group=manifest["campaign_id"],
            cyp="all",
            config={"fold": manifest["config"]["fold"]},
            mode=mode,
        )
        tracker.log(values)
        tracker.finish(values)
    except Exception as exc:
        # Remote summary trouble must not lose an already completed trial or promotion.
        (output / "tracking-error.json").write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "leaderboard_preserved": True,
                }
            )
        )
