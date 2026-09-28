"""Small, inspectable baseline model families."""

from __future__ import annotations

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REGRESSION_MODELS = ("ridge", "extra_trees", "lightgbm")
CLASSIFICATION_MODELS = ("logistic", "extra_trees", "lightgbm")


def make_regressor(name: str, seed: int = 42):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown regression model {name!r}; choose from {REGRESSION_MODELS}")


def make_classifier(name: str, seed: int = 42):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown classifier {name!r}; choose from {CLASSIFICATION_MODELS}")
