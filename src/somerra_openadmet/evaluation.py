"""Shared evaluation dispatch for research trials and saved-recipe reproduction.

This module does not choose experiments or promote recipes. Both workflows use
the same backend implementations, completion checks and primary score.
"""

import math


def evaluate_recipe(*, backend, model_config, data_config, output_dir):
    if backend == "cpu":
        from .campaign import evaluate
    elif backend == "neural":
        from .neural import screen as evaluate
    else:
        raise ValueError("backend must be cpu or neural")
    result = evaluate(model_config=model_config, data_config=data_config, output_dir=output_dir)
    if backend == "neural" and result.get("complete") is not True:
        raise ValueError("Incomplete neural trial cannot receive a research score")
    score = result["macro"]["nested_ensemble"] if backend == "cpu" else result["MA-ST-RAE"]
    if not math.isfinite(score):
        raise ValueError("Non-finite research score")
    return result, score
