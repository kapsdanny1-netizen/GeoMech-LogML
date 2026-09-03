"""Model registry: Random Forest, Gradient Boosting (XGBoost), shallow MLP.

Design decisions (see ADR.md for the full rationale):

* One estimator **per target** (E_STAT, NU_STAT, UCS). This keeps SHAP, quantile
  intervals and hyper-parameters per target simple and honest.
* Trees get unscaled features (scale-invariant); the MLP is wrapped in a
  ``Pipeline(StandardScaler, MLPRegressor)``.
* The MLP uses ``early_stopping=False`` deliberately — sklearn's internal early
  stopping uses a **random** validation split, which would violate the
  "no random splits" rule inside our well-wise CV.
* Every estimator takes an explicit ``random_state`` → full reproducibility.
* No Vs-derived feature exists anywhere: static↔dynamic conversion is learned
  implicitly by the model from the logs (requirement 5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from geomech_logml.config import GLOBAL_SEED, TARGETS

__all__ = ["MODEL_SPECS", "build_model", "train_models", "DEFAULT_MODELS"]


# ---------------------------------------------------------------------------
# Hyper-parameter defaults (modest, robust, CPU-friendly)
# ---------------------------------------------------------------------------
RF_PARAMS: dict = dict(
    n_estimators=300,
    min_samples_leaf=2,
    max_features=0.6,
    n_jobs=-1,
)
XGB_PARAMS: dict = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    objective="reg:squarederror",
)
MLP_PARAMS: dict = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    alpha=1e-3,
    learning_rate_init=1e-3,
    max_iter=3000,
    early_stopping=False,   # NOTE: random internal split would violate well-wise CV
    tol=1e-5,
)


@dataclass
class ModelSpec:
    """Description + factory for one model family."""
    key: str
    label: str
    family: str                      # "rf" | "xgb" | "mlp"
    shap_kind: str                   # "tree" | "kernel"
    supports_qrf: bool               # native quantile extraction from leaves
    builder: Callable[..., object]   # builder(random_state=..., **overrides)

    def build(self, seed: int = GLOBAL_SEED, overrides: dict | None = None) -> object:
        params = dict(RF_PARAMS if self.family == "rf" else
                      XGB_PARAMS if self.family == "xgb" else MLP_PARAMS)
        if overrides:
            params.update({k: v for k, v in overrides.items() if v is not None})
        return self.builder(random_state=seed, **params)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _build_rf(random_state: int, **kw) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=random_state, **kw)


def _build_xgb(random_state: int, **kw) -> XGBRegressor:
    # silence the JSON-config verbosity, keep determinism
    kw.setdefault("verbosity", 0)
    return XGBRegressor(random_state=random_state, **kw)


def _build_mlp(random_state: int, **kw) -> Pipeline:
    mlp = MLPRegressor(random_state=random_state, **kw)
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


MODEL_SPECS: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        key="random_forest", label="Random Forest", family="rf",
        shap_kind="tree", supports_qrf=True, builder=_build_rf),
    "xgboost": ModelSpec(
        key="xgboost", label="XGBoost (Gradient Boosting)", family="xgb",
        shap_kind="tree", supports_qrf=False, builder=_build_xgb),
    "mlp": ModelSpec(
        key="mlp", label="Shallow MLP (sklearn)", family="mlp",
        shap_kind="kernel", supports_qrf=False, builder=_build_mlp),
}

DEFAULT_MODELS: list[str] = ["random_forest", "xgboost", "mlp"]


def build_model(key: str, seed: int = GLOBAL_SEED, overrides: dict | None = None):
    """Instantiate one model by registry key (with hyper-parameter overrides)."""
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model '{key}'. Choose from {list(MODEL_SPECS)}")
    return MODEL_SPECS[key].build(seed=seed, overrides=overrides)


# ---------------------------------------------------------------------------
# Training (one estimator per target)
# ---------------------------------------------------------------------------
def train_models(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    model_keys: list[str] | None = None,
    seed: int = GLOBAL_SEED,
    overrides: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, object]]:
    """Fit every requested model on the (core-calibrated) training rows.

    Parameters
    ----------
    X, Y : feature matrix and multi-target frame (aligned indexes).
    model_keys : subset of MODEL_SPECS to train (default: all three).
    seed : global reproducibility seed.
    overrides : optional per-model hyper-parameter overrides,
        e.g. ``{"random_forest": {"n_estimators": 100}}``.

    Returns
    -------
    ``{model_key: {target: fitted_estimator}}``
    """
    model_keys = model_keys or DEFAULT_MODELS
    overrides = overrides or {}
    fitted: dict[str, dict[str, object]] = {}

    for key in model_keys:
        spec = MODEL_SPECS[key]
        fitted[key] = {}
        t0 = time.perf_counter()
        for target in TARGETS:
            est = spec.build(seed=seed, overrides=overrides.get(key))
            est.fit(X, Y[target].to_numpy())
            fitted[key][target] = est
        if verbose:
            print(f"  fitted {spec.label:28s} in {time.perf_counter() - t0:5.1f}s")
    return fitted


def predict_targets(
    model_bundle: dict[str, object],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Predict all targets with a per-target model dict → tidy DataFrame."""
    return pd.DataFrame(
        {t: model_bundle[t].predict(X) for t in TARGETS},
        index=X.index,
    )
