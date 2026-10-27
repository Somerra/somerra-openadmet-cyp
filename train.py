"""Editable model recipe for the frozen regression task; use autoresearch.py plan/run."""

from pathlib import Path

MODEL_RECIPE = Path(__file__).parent / "configs/regression/incumbent.json"
MODEL_OVERRIDES = {}  # Propose one model change here before plan/run.


def main():
    from somerra_openadmet.regression_task import main as task_main

    return task_main(default_command="run")


if __name__ == "__main__":
    main()
