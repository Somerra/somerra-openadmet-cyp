"""Canonical research CLI: plan without training, or run one explicit trial.

Edit train.py for experiments; keep this launcher and the evaluator fixed.
"""

from pathlib import Path

import train
from somerra_openadmet.research import main

if __name__ == "__main__":
    main(
        backend=train.BACKEND,
        model_recipe=train.MODEL_RECIPE,
        overrides=train.MODEL_OVERRIDES,
        recipe_source=Path(train.__file__),
    )
