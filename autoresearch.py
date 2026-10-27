"""Primary interface for autoresearch trials and saved-recipe reproduction.

Edit train.py for experiments; keep this launcher and the evaluator fixed.
"""

import sys
from pathlib import Path

from somerra_openadmet.research import main as run_command


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "hillclimb":
        from somerra_openadmet.hillclimb import main as hillclimb_main

        return hillclimb_main(argv[1:])
    if argv and argv[0] == "reference":
        from somerra_openadmet.reference import main as reference_main

        return reference_main(argv[1:])
    if argv and argv[0] == "reproduce":
        # Reproduction must not execute the mutable experiment recipe or inherit
        # its overrides, including when train.py is temporarily broken mid-edit.
        return run_command(
            backend=None, model_recipe=None, overrides={}, recipe_source=None, argv=argv
        )
    import train

    return run_command(
        backend=train.BACKEND,
        model_recipe=train.MODEL_RECIPE,
        overrides=train.MODEL_OVERRIDES,
        recipe_source=Path(train.__file__),
        argv=argv,
    )


if __name__ == "__main__":
    main()
