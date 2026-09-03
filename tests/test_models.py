"""Model training + well-wise CV evaluation tests (small, fast settings)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomech_logml.config import TARGETS
from geomech_logml.models.evaluate import metrics_table, run_well_wise_cv
from geomech_logml.models.registry import (
    DEFAULT_MODELS,
    build_model,
    train_models,
)
from geomech_logml.preprocessing.cv import WellKFold


Mlp_Fast = {"max_iter": 300, "hidden_layer_sizes": (32, 16)}


def _overrides(key):
    return Mlp_Fast if key == "mlp" else None


def test_registry_builds_all_families():
    for key in DEFAULT_MODELS:
        est = build_model(key, seed=0)
        assert hasattr(est, "fit") and hasattr(est, "predict")


def test_mlp_has_no_random_early_stopping():
    """sklearn MLP early stopping uses a random split — must be disabled."""
    pipe = build_model("mlp")
    assert pipe.named_steps["mlp"].early_stopping is False


def test_train_models_fits_all_targets(core_matrices):
    X, Y, g = core_matrices
    sub = np.random.default_rng(0).choice(len(X), size=min(150, len(X)), replace=False)
    bundle = train_models(X.iloc[sub], Y.iloc[sub], model_keys=DEFAULT_MODELS,
                          overrides={"mlp": Mlp_Fast})
    for key in DEFAULT_MODELS:
        for target in ("E_STAT", "NU_STAT", "UCS"):
            assert key in bundle and target in bundle[key]


def test_well_wise_cv_random_forest_learns(core_matrices):
    """Blind-well performance must be strong on synthetic data (known physics)."""
    X, Y, g = core_matrices
    res = run_well_wise_cv(X, Y, g, "random_forest",
                           splitter=WellKFold(3, seed=0), seed=0)
    pooled = res.pooled_metrics.set_index("TARGET")
    assert pooled.loc["E_STAT", "R2"] > 0.7
    assert pooled.loc["NU_STAT", "R2"] > 0.6
    assert pooled.loc["UCS", "R2"] > 0.5
    # every core row predicted exactly once
    assert len(res.oof) == len(X) * 3


def test_oof_uses_blind_wells_only(core_matrices):
    X, Y, g = core_matrices
    res = run_well_wise_cv(X, Y, g, "xgboost", splitter=WellKFold(3, seed=0), seed=0)
    for fold, sub in res.oof.groupby("FOLD"):
        test_wells = set(sub["WELL"])
        fold_metrics = res.fold_metrics[res.fold_metrics["FOLD"] == fold]
        assert set(fold_metrics["TEST_WELLS"].iloc[0].split(",")) == test_wells


def test_metrics_table_shape(core_matrices):
    X, Y, g = core_matrices
    res = run_well_wise_cv(X, Y, g, "mlp", splitter=WellKFold(3, seed=0), seed=0,
                           overrides=Mlp_Fast)
    table = metrics_table({"mlp": res})
    assert set(table.columns) == {"Model", "ModelKey", "Target", "R2", "RMSE", "MAE"}
    assert len(table) == 3


def test_predictions_are_physical(core_matrices):
    """Pipeline clipping keeps predictions inside target bounds."""
    from geomech_logml.config import TARGET_BOUNDS
    X, Y, g = core_matrices
    res = run_well_wise_cv(X, Y, g, "random_forest", splitter=WellKFold(3, seed=0), seed=0)
    preds = pd.DataFrame(
        {t: res.final_models[t].predict(X) for t in TARGETS})
    for t, (lo, hi) in TARGET_BOUNDS.items():
        assert preds[t].min() >= lo - 1e-6
        assert preds[t].max() <= hi + 1e-6
