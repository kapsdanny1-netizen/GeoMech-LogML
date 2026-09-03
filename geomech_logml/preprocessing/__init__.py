"""Preprocessing: log cleaning, petrophysical feature engineering, feature sets.

Design rule: every engineered feature is derived **only** from the standard/legacy
input suite (GR, RHOB, NPHI, RT and optionally VP/DEPT). No shear-sonic dependent
quantity is ever computed or required.
"""

from geomech_logml.preprocessing.features import (
    clean_logs,
    engineer_features,
    FEATURE_SETS,
    get_feature_names,
    build_feature_matrix,
)
from geomech_logml.preprocessing.cv import WellKFold, LeaveOneWellOut, make_splitter

__all__ = [
    "clean_logs",
    "engineer_features",
    "FEATURE_SETS",
    "get_feature_names",
    "build_feature_matrix",
    "WellKFold",
    "LeaveOneWellOut",
    "make_splitter",
]
