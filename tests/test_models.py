import numpy as np

from somerra_openadmet.models import make_classifier, make_regressor


def test_all_model_families_fit_small_problem():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 12))
    regression_target = features[:, 0] - 0.5 * features[:, 1]
    classification_target = (regression_target > np.median(regression_target)).astype(int)

    for name in ("ridge", "extra_trees", "lightgbm"):
        prediction = (
            make_regressor(name, seed=7, n_jobs=1)
            .fit(features, regression_target)
            .predict(features)
        )
        assert prediction.shape == (40,)
        assert np.isfinite(prediction).all()

    for name in ("logistic", "extra_trees", "lightgbm"):
        probability = (
            make_classifier(name, seed=7, n_jobs=1)
            .fit(features, classification_target)
            .predict_proba(features)[:, 1]
        )
        assert probability.shape == (40,)
        assert ((probability >= 0.0) & (probability <= 1.0)).all()
