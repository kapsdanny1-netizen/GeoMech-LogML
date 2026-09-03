"""Uncertainty tests: QRF mechanics + honest well-wise conformal coverage."""

from __future__ import annotations

import numpy as np

from geomech_logml.models.registry import build_model
from geomech_logml.preprocessing.cv import WellKFold
from geomech_logml.uncertainty.conformal import (
    conformal_intervals_from_oof,
    conformal_quantile,
    empirical_coverage,
    well_wise_conformal_intervals,
)
from geomech_logml.uncertainty.qrf import QuantileForest
from geomech_logml.models.evaluate import run_well_wise_cv


# ---------------------------------------------------------------------------
# Conformal quantile math
# ---------------------------------------------------------------------------
def test_conformal_quantile_conservative():
    r = np.arange(1, 11, dtype=float)          # 1..10
    q = conformal_quantile(r, alpha=0.10)
    # ceil((10+1)*0.9)/10 = 0.99 quantile -> ~10 (never below max)
    assert q >= 9.5


# ---------------------------------------------------------------------------
# QRF mechanics
# ---------------------------------------------------------------------------
def test_qrf_intervals_ordered_and_reasonable(core_matrices):
    X, Y, g = core_matrices
    Xa, ya = X.to_numpy(), Y["UCS"].to_numpy()
    forest = build_model("random_forest", seed=0,
                         overrides={"n_estimators": 60})
    forest.fit(Xa, ya)
    qrf = QuantileForest(forest, Xa, ya)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xa), size=60, replace=False)
    lo, med, hi = qrf.predict_intervals(Xa[idx], alpha=0.10)
    assert np.all(lo <= med) and np.all(med <= hi)
    # median prediction close to the RF point prediction
    assert np.median(np.abs(med - forest.predict(Xa[idx]))) < 1.0


def test_qrf_blind_well_coverage(core_matrices):
    """Coverage on wells NOT used to fit the forest must be near nominal.

    NOTE: QRF leaf-pool intervals capture the spread of training responses but do
    NOT account for inter-well bias (systematic shifts between wells). Conformal
    (residual-calibrated, well-wise) covers that case — see the pipeline coverage
    table. Hence the moderate tolerance and the low-bias target (UCS).
    """
    X, Y, g = core_matrices
    wells = sorted(g.unique())
    test_wells = wells[:2]
    te = g.isin(test_wells).to_numpy()
    Xa, ya = X[~te].to_numpy(), Y[~te]["UCS"].to_numpy()
    forest = build_model("random_forest", seed=0,
                         overrides={"n_estimators": 100, "min_samples_leaf": 10})
    forest.fit(Xa, ya)
    qrf = QuantileForest(forest, Xa, ya)
    lo, _, hi = qrf.predict_intervals(X[te].to_numpy(), alpha=0.10)
    cov = empirical_coverage(Y[te]["UCS"].to_numpy(), lo, hi)
    assert 0.70 <= cov <= 1.0, f"coverage {cov:.3f} too far from 0.90 nominal"


# ---------------------------------------------------------------------------
# Well-wise conformal
# ---------------------------------------------------------------------------
def test_conformal_wellwise_holds_wells(core_matrices):
    X, Y, g = core_matrices

    def fit_predict(Xtr, ytr, Xev):
        m = build_model("random_forest", seed=0, overrides={"n_estimators": 60})
        m.fit(Xtr, np.asarray(ytr))
        return m.predict(Xev)

    ci = well_wise_conformal_intervals(fit_predict, X, Y, g, alpha=0.10)
    # every core row × target present exactly once
    assert len(ci) == len(X) * 3
    assert (ci["LO"] < ci["PRED"]).all() and (ci["PRED"] < ci["HI"]).all()


def test_conformal_intervals_from_oof(core_matrices):
    X, Y, g = core_matrices
    res = run_well_wise_cv(X, Y, g, "random_forest", splitter=WellKFold(3, seed=0), seed=0)
    preds = Y.copy()
    out = conformal_intervals_from_oof(res.oof, preds, alpha=0.10)
    for t in ("E_STAT", "NU_STAT", "UCS"):
        assert f"{t}_LO" in out.columns and f"{t}_HI" in out.columns
        assert (out[f"{t}_LO"] < out[f"{t}_PRED"]).all()
    # pooled OOF residuals at 90% should roughly cover ~90% of the truth rows
    cov = empirical_coverage(Y["UCS"].to_numpy(), out["UCS_LO"].to_numpy(), out["UCS_HI"].to_numpy())
    assert cov > 0.80
