"""Primary interface for autoresearch trials and saved-recipe reproduction.

Edit train.py for experiments; keep this launcher and the evaluator fixed.
"""

import sys
from pathlib import Path

from somerra_openadmet.research import main as run_command


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "single":
        from somerra_openadmet.single_model import main as single_main

        return single_main(argv[1:])
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

    return task_main(argv)


if __name__ == "__main__":
    main()
