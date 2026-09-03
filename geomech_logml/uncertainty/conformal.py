"""Split-conformal prediction intervals with **well-wise** calibration.

Conformal prediction gives finite-sample coverage guarantees if calibration
residuals are exchangeable with test points. In spatial earth science they are
NOT (adjacent depths / same-well samples are correlated), so we calibrate in the
only defensible way here: calibration residuals always come from wells the model
was **not** trained on.

Two entry points
----------------
* :func:`well_wise_conformal_intervals` — nested, strictly honest: within each CV
  fold, one *calibration well* is carved out of the training wells; its absolute
  residuals give the interval width applied to the held-out test well. Used for
  coverage validation.
* :func:`conformal_intervals_from_oof` — practical: quantiles of pooled
  out-of-fold residuals (from well-wise CV) are applied as ± widths around the
  final model's predictions. Used for live curve prediction in the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomech_logml.config import TARGETS

__all__ = [
    "well_wise_conformal_intervals",
    "conformal_intervals_from_oof",
    "empirical_coverage",
    "conformal_quantile",
]


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of |residuals| at level 1 − alpha."""
    r = np.sort(np.asarray(residuals, dtype=float))
    n = r.size
    if n == 0:
        return np.nan
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)   # conservative step-up
    return float(np.quantile(r, level, method="higher"))


def well_wise_conformal_intervals(
    fit_predict: callable,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    groups: pd.Series,
    alpha: float = 0.10,
    n_calib_wells: int = 2,
) -> pd.DataFrame:
    """Nested well-wise split-conformal intervals for every training row.

    Within each rotation, ``n_calib_wells`` training wells (default 2 — pooling
    stabilises the residual quantile against single-well quirks) are reserved for
    calibration; the model fits the rest and predicts the held-out test well.

    Parameters
    ----------
    fit_predict : function(X_train, y_train, X_eval) -> np.ndarray predictions
    X, Y, groups : aligned features / targets / well ids (core rows).
    alpha : miscoverage rate (interval target = 1 − alpha).
    n_calib_wells : calibration wells per rotation (capped at len(train_wells) − 2).

    Returns
    -------
    DataFrame with columns WELL, ROW, TARGET, TRUE, PRED, LO, HI, WIDTH.
    """
    wells = groups.to_numpy()
    unique_wells = sorted(np.unique(wells))
    if len(unique_wells) < 4:
        raise ValueError("Nested well-wise conformal needs >= 4 wells "
                         "(1 test + 2 calibration + >=1 fit).")
    records: list[dict] = []

    for i_w, test_well in enumerate(unique_wells):
        test_idx = np.where(wells == test_well)[0]
        train_wells = [w for w in unique_wells if w != test_well]
        n_cal = min(n_calib_wells, max(1, len(train_wells) - 2))
        # deterministic rotation of calibration wells
        calib_wells = [train_wells[(i_w + k) % len(train_wells)] for k in range(n_cal)]
        fit_wells = [w for w in train_wells if w not in calib_wells]

        fit_idx = np.where(np.isin(wells, fit_wells))[0]
        calib_idx = np.where(np.isin(wells, calib_wells))[0]

        for target in TARGETS:
            preds_cal = fit_predict(X.iloc[fit_idx], Y.iloc[fit_idx][target], X.iloc[calib_idx])
            resid = np.abs(Y.iloc[calib_idx][target].to_numpy() - preds_cal)
            width = conformal_quantile(resid, alpha)

            preds_test = fit_predict(X.iloc[fit_idx], Y.iloc[fit_idx][target], X.iloc[test_idx])
            truth = Y.iloc[test_idx][target].to_numpy()
            records.extend(
                {"WELL": wells[j], "ROW": int(j), "TARGET": target,
                 "TRUE": float(truth[k]), "PRED": float(preds_test[k]),
                 "LO": float(preds_test[k] - width), "HI": float(preds_test[k] + width),
                 "WIDTH": float(2 * width)}
                for k, j in enumerate(test_idx)
            )

    return pd.DataFrame(records)


def conformal_intervals_from_oof(
    oof: pd.DataFrame,
    predictions: pd.DataFrame,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Attach ± conformal widths (from pooled well-wise OOF residuals) to final
    model predictions.

    Parameters
    ----------
    oof : long-format OOF frame with columns TARGET, TRUE, PRED
        (as produced by ``run_well_wise_cv``).
    predictions : wide frame of final predictions, one column per target
        (index aligned with the prediction rows).
    alpha : miscoverage rate.

    Returns
    -------
    DataFrame with columns {target}_PRED, {target}_LO, {target}_HI plus WIDTH_* columns.
    """
    out = pd.DataFrame(index=predictions.index)
    for target in TARGETS:
        sub = oof[oof["TARGET"] == target]
        if sub.empty:
            raise ValueError(f"OOF frame has no rows for target {target}")
        width = conformal_quantile((sub["TRUE"] - sub["PRED"]).abs().to_numpy(), alpha)
        out[f"{target}_PRED"] = predictions[target]
        out[f"{target}_LO"] = predictions[target] - width
        out[f"{target}_HI"] = predictions[target] + width
        out[f"{target}_WIDTH"] = 2 * width
    return out


def empirical_coverage(
    y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    """Fraction of observations inside [lo, hi]."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))
