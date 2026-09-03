"""Well-wise out-of-fold (OOF) training & evaluation.

The evaluation loop respects the hard validation rule: each fold holds out whole
wells; the model never sees any row of a held-out well during training. Metrics are
computed per fold *and* pooled over all out-of-fold predictions (an honest estimate
of blind-well performance).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from geomech_logml.config import GLOBAL_SEED, TARGETS, WELL_COL
from geomech_logml.models.registry import MODEL_SPECS, train_models
from geomech_logml.preprocessing.cv import WellKFold

__all__ = ["CVResult", "run_well_wise_cv", "metrics_table", "summarise_metrics"]


@dataclass
class CVResult:
    """Out-of-fold predictions + fitted final models for one experiment."""
    model_key: str
    oof: pd.DataFrame              # WELL, FOLD, {target}_TRUE, {target}_PRED
    fold_metrics: pd.DataFrame     # per (fold, target) metrics
    pooled_metrics: pd.DataFrame   # per target metrics over pooled OOF
    final_models: dict             # {target: estimator} trained on ALL core rows
    fit_seconds: float


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def run_well_wise_cv(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    groups: pd.Series,
    model_key: str,
    splitter=None,
    seed: int = GLOBAL_SEED,
    overrides: dict | None = None,
) -> CVResult:
    """Train/evaluate one model family under well-wise CV and refit on all rows.

    Parameters
    ----------
    X, Y, groups : features, targets, well ids (aligned).
    model_key : one of ``MODEL_SPECS``.
    splitter : WellKFold / LeaveOneWellOut instance (default WellKFold, k=min(5, wells)).
    """
    if splitter is None:
        splitter = WellKFold(n_splits=min(5, groups.nunique()), seed=seed)

    wells = groups.to_numpy()
    oof_rows: list[dict] = []
    fold_records: list[dict] = []
    t0 = time.perf_counter()

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, groups=wells)):
        # per-target estimators per fold
        for target in TARGETS:
            est = MODEL_SPECS[model_key].build(seed=seed, overrides=overrides)
            est.fit(X.iloc[train_idx], Y.iloc[train_idx][target].to_numpy())
            pred = est.predict(X.iloc[test_idx])
            for row_i, p in zip(test_idx, pred):
                oof_rows.append(
                    {"ROW": row_i, WELL_COL: wells[row_i], "FOLD": fold,
                     "TARGET": target,
                     "TRUE": float(Y.iloc[row_i][target]),
                     "PRED": float(p)}
                )
            m = _score(Y.iloc[test_idx][target].to_numpy(), pred)
            fold_records.append({"FOLD": fold, "TARGET": target, **m,
                                 "TEST_WELLS": ",".join(sorted(set(wells[test_idx])))})

    fit_seconds = time.perf_counter() - t0
    oof = pd.DataFrame(oof_rows)

    pooled = []
    for target in TARGETS:
        sub = oof[oof["TARGET"] == target]
        pooled.append({"TARGET": target, **_score(sub["TRUE"].to_numpy(), sub["PRED"].to_numpy())})
    pooled_df = pd.DataFrame(pooled)

    # Final models trained on every core row (used for live curve prediction).
    final = train_models(X, Y, model_keys=[model_key], seed=seed, overrides=overrides)[model_key]

    return CVResult(
        model_key=model_key,
        oof=oof,
        fold_metrics=pd.DataFrame(fold_records),
        pooled_metrics=pooled_df,
        final_models=final,
        fit_seconds=fit_seconds,
    )


def metrics_table(results: dict[str, CVResult]) -> pd.DataFrame:
    """Side-by-side pooled OOF metrics: rows = model × target."""
    rows = []
    for key, res in results.items():
        for _, r in res.pooled_metrics.iterrows():
            rows.append({
                "Model": MODEL_SPECS[key].label,
                "ModelKey": key,
                "Target": r["TARGET"],
                "R2": round(r["R2"], 3),
                "RMSE": round(r["RMSE"], 3),
                "MAE": round(r["MAE"], 3),
            })
    return pd.DataFrame(rows)


def summarise_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Pivot the metrics table to Model × Target cells of R² (RMSE) for reports."""
    if table.empty:
        return table
    def cell(r):
        return f"{r['R2']:+.3f} ({r['RMSE']:.2f})"
    return (
        table.assign(CELL=table.apply(cell, axis=1))
        .pivot(index="Model", columns="Target", values="CELL")
    )
