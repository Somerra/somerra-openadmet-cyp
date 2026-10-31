"""Primary interface for autoresearch trials and saved-recipe reproduction.

CYP recipes are the default; historical workflows remain explicit subcommands.
"""

import sys
from pathlib import Path

from somerra_openadmet.research import main as run_command


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"recipes", "prepare", "trial", "status", "note", "--help", "-h"}:
        from somerra_openadmet.cyp_research import main as cyp_main

        return cyp_main(argv[1:] if argv and argv[0] == "recipes" else argv)
    if argv and argv[0] == "briford":
        from somerra_openadmet.briford import main as briford_main

        return briford_main(argv[1:])
    if argv and argv[0] == "scaffold":
        from somerra_openadmet.scaffold_ensemble import main as scaffold_main

        return scaffold_main(argv[1:])
    if argv and argv[0] == "hillclimb":
        from somerra_openadmet.hillclimb import main as hillclimb_main

        return hillclimb_main(argv[1:])
    if argv and argv[0] == "reference":
        from somerra_openadmet.reference import main as reference_main

        return reference_main(argv[1:])
    if argv and argv[0] == "legacy":
        if len(argv) > 1 and argv[1] == "reproduce":
            return run_command(
                backend=None, model_recipe=None, overrides={}, recipe_source=None, argv=argv[1:]
            )
        import legacy_train

        return run_command(
            backend=legacy_train.BACKEND,
            model_recipe=legacy_train.MODEL_RECIPE,
            overrides=legacy_train.MODEL_OVERRIDES,
            recipe_source=Path(legacy_train.__file__),
            argv=argv[1:],
        )
    from somerra_openadmet.regression_task import main as task_main

    if argv and argv[0] == "molecular":
        return task_main(argv[1:])
    if argv and argv[0] in {"plan", "run"}:
        print(
            "WARNING: bare autoresearch.py plan/run uses the historical 60/20/20 "
            "molecular task, not the default Workbench scaffold protocol. "
            "See docs/workbench-protocol.md. Use 'autoresearch.py molecular "
            f"{argv[0]}' to select the historical task explicitly, or "
            "'autoresearch.py briford' for the implemented Workbench reference.",
            file=sys.stderr,
        )
    return task_main(argv)


if __name__ == "__main__":
    main()
