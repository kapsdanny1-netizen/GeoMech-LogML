"""Trained-model persistence: export/import joblib bundles + prediction-only mode.

A bundle contains everything needed to apply a trained model to new wells without
retraining:

* the per-target fitted estimators for one model family,
* the exact feature list (and feature set name) the model expects,
* the conformal interval half-widths (per target) at export time,
* metadata (model key, alpha, seed, versions, creation timestamp).

Workflow in the app:  Train & validate → ⬇️ Download model (.joblib) → later,
upload the file, press "Predict with loaded model" and get full-curve predictions
with interval bands on the current data — no cross-validation, no retraining.
"""

from __future__ import annotations

import datetime as _dt
import io

import joblib
import numpy as np
import pandas as pd

from geomech_logml.config import (
    CORE_FLAG,
    DEPTH_COL,
    GLOBAL_SEED,
    TARGETS,
    TARGET_BOUNDS,
    WELL_COL,
)
from geomech_logml.models.registry import MODEL_SPECS
from geomech_logml.pipeline import ExperimentResult
from geomech_logml.preprocessing.features import (
    clean_logs,
    engineer_features,
)

__all__ = [
    "bundle_from_result",
    "bundle_to_bytes",
    "bundle_from_bytes",
    "predict_with_bundle",
    "result_from_bundle",
]

BUNDLE_VERSION = 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def bundle_from_result(result: ExperimentResult, model_key: str) -> dict:
    """Assemble a serialisable bundle for one model family of a finished experiment."""
    if model_key not in result.final_models:
        raise KeyError(f"Model '{model_key}' was not trained in this experiment.")
    widths = {}
    for target in TARGETS:
        oof = result.cv[model_key].oof
        sub = oof[oof["TARGET"] == target]
        from geomech_logml.uncertainty.conformal import conformal_quantile
        widths[target] = conformal_quantile(
            (sub["TRUE"] - sub["PRED"]).abs().to_numpy(), result.config.alpha)
    return {
        "format": "geomech-logml-bundle",
        "bundle_version": BUNDLE_VERSION,
        "model_key": model_key,
        "model_label": MODEL_SPECS[model_key].label,
        "feature_set": result.config.feature_set,
        "feature_names": list(result.feature_names),
        "alpha": result.config.alpha,
        "seed": result.config.seed,
        "conformal_widths": widths,
        "models": dict(result.final_models[model_key]),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "package_version": _pkg_version(),
    }


def _pkg_version() -> str:
    try:
        from geomech_logml import __version__
        return __version__
    except Exception:  # pragma: no cover
        return "unknown"


def bundle_to_bytes(bundle: dict) -> bytes:
    """Serialise a bundle to raw bytes (for a Streamlit download button)."""
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def bundle_from_bytes(raw: bytes | io.BytesIO) -> dict:
    """Deserialise a bundle produced by :func:`bundle_to_bytes`."""
    src = raw if isinstance(raw, io.BytesIO) else io.BytesIO(raw)
    try:
        bundle = joblib.load(src)
    except Exception as e:  # noqa: BLE001 — corrupted/foreign file
        raise ValueError(f"Could not read a GeoMech-LogML bundle from this file ({e}).")
    if not isinstance(bundle, dict) or bundle.get("format") != "geomech-logml-bundle":
        raise ValueError("This file is not a GeoMech-LogML model bundle.")
    missing = [t for t in TARGETS if t not in bundle.get("models", {})]
    if missing:
        raise ValueError(f"Bundle is missing estimators for: {missing}")
    return bundle


# ---------------------------------------------------------------------------
# Prediction with a bundle
# ---------------------------------------------------------------------------
def predict_with_bundle(df_raw: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a loaded model over a raw log frame (no training, no CV).

    Returns
    -------
    (curves, data_engineered) : the prediction frame (WELL, DEPT, per-target
    PRED/LO/HI columns named ``{target}_{model_key}`` like a normal run) and the
    engineered frame it was computed from (for plots / SHAP).
    """
    data = engineer_features(clean_logs(df_raw))
    feats = [f for f in bundle["feature_names"] if f in data.columns]
    missing = [f for f in bundle["feature_names"] if f not in data.columns]
    if missing:
        raise ValueError(
            f"Loaded model needs features that this dataset cannot provide: {missing}. "
            f"Upload logs covering them (e.g. Vp/DTC for the 'with Vp' feature sets).")
    mask = data[feats].notna().all(axis=1)
    sub = data.loc[mask]
    Xc = sub[feats].astype(float)

    key = bundle["model_key"]
    out = sub[[WELL_COL, DEPTH_COL]].copy()
    if CORE_FLAG in sub.columns:
        out[CORE_FLAG] = sub[CORE_FLAG].to_numpy()
    for truth_col in TARGETS:
        if truth_col in sub.columns:
            out[f"{truth_col}_TRUE"] = sub[truth_col].to_numpy()
    for facies_col in ("FACIES", "VSH_TRUE"):
        if facies_col in sub.columns:
            out[facies_col] = sub[facies_col].to_numpy()
    for target in TARGETS:
        pred = bundle["models"][target].predict(Xc)
        lo_b, hi_b = TARGET_BOUNDS[target]
        pred = np.clip(pred, lo_b, hi_b)
        width = float(bundle["conformal_widths"].get(target, 0.0))
        out[f"{target}_{key}"] = pred
        out[f"{target}_{key}_LO"] = pred - width
        out[f"{target}_{key}_HI"] = pred + width
    return out.reset_index(drop=True), data


def result_from_bundle(
    df_raw: pd.DataFrame,
    bundle: dict,
    max_explain_rows: int = 500,
) -> ExperimentResult:
    """Build a prediction-only ``ExperimentResult`` for the dashboard.

    Cross-validation fields (metrics, oof, interval validation) are empty by
    design — there was no training. SHAP works against the predicted rows.
    """
    curves, data = predict_with_bundle(df_raw, bundle)
    key = bundle["model_key"]

    feats = list(bundle["feature_names"])
    mask = data[feats].notna().all(axis=1)
    X_pred = data.loc[mask, feats].astype(float)
    if len(X_pred) > max_explain_rows:      # subsample for SHAP background speed
        X_pred = X_pred.iloc[np.linspace(0, len(X_pred) - 1, max_explain_rows).astype(int)]

    from geomech_logml.pipeline import ExperimentConfig
    cfg = ExperimentConfig(
        feature_set=bundle.get("feature_set", "custom"),
        model_keys=[key], cv_strategy="none", n_splits=2,
        alpha=float(bundle.get("alpha", 0.10)), seed=int(bundle.get("seed", GLOBAL_SEED)),
        validate_uncertainty=False)

    return ExperimentResult(
        config=cfg, data=data,
        X_core=X_pred.reset_index(drop=True),
        Y_core=pd.DataFrame(columns=TARGETS),
        groups=pd.Series(name=WELL_COL, dtype=object),
        feature_names=feats,
        cv={}, metrics=pd.DataFrame(columns=["Model", "ModelKey", "Target", "R2", "RMSE", "MAE"]),
        honest_conformal=None, qrf_oof=None,
        curves=curves, final_models={key: bundle["models"]},
        runtime_seconds=0.0,
    )
