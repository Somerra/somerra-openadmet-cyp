"""Editable regression experiment recipe; the evaluator lives in src/ and stays fixed.

See program.md before running. Each invocation runs ONE experiment, never a loop.
"""

from pathlib import Path

from somerra_openadmet.research import main as research_main

# Model research changes model knobs; data research changes only data_sources.
BACKEND = "cpu"  # "cpu" (nested evaluation) or "neural" (screening only)
MODEL_RECIPE = Path("configs/models/incumbent.json")
MODEL_OVERRIDES = {}


def main():
    """Compatibility launcher; prefer autoresearch.py plan/run for new trials."""
    return research_main(
        backend=BACKEND,
        model_recipe=MODEL_RECIPE,
        overrides=MODEL_OVERRIDES,
        recipe_source=Path(__file__),
        default_command="run",
    )


if __name__ == "__main__":
    main()
