"""Uncertainty quantification: prediction intervals for every prediction.

Two complementary, literature-backed methods are provided:

* **QRF** — Quantile Regression Forests (Meinshausen, 2006): exact quantiles of the
  training responses pooled over RF leaf memberships. Native to Random Forests.
* **Conformal prediction** (split-conformal, model-agnostic): finite-sample
  calibrated residual quantiles — computed **well-wise**, i.e. the calibration
  residuals always come from held-out wells, never from the well being predicted.

Both return per-row lower/upper bounds for each target at a chosen confidence
level (default 90%). Empirical coverage is validated in the test suite.
"""

from geomech_logml.uncertainty.qrf import QuantileForest
from geomech_logml.uncertainty.conformal import (
    well_wise_conformal_intervals,
    conformal_intervals_from_oof,
    empirical_coverage,
)

__all__ = [
    "QuantileForest",
    "well_wise_conformal_intervals",
    "conformal_intervals_from_oof",
    "empirical_coverage",
]
