"""Feature engineering + well-wise CV integrity tests."""

from __future__ import annotations

import numpy as np
import pytest

from geomech_logml.preprocessing.cv import LeaveOneWellOut, WellKFold, make_splitter
from geomech_logml.preprocessing.features import (
    FEATURE_SETS,
    clean_logs,
    get_feature_names,
)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_engineered_features_finite(engineered):
    feat_cols = ["RT_LOG10", "VSH_GR", "PHID", "PHIND", "VPHI_RATIO", "PHIS", "AI"]
    for c in feat_cols:
        if c in engineered.columns:
            assert np.isfinite(engineered[c].to_numpy()).all(), f"{c} has non-finite values"


def test_vsh_gr_in_unit_interval(engineered):
    assert engineered["VSH_GR"].between(0, 1).all()


def test_feature_sets_resolve(engineered):
    for fs in FEATURE_SETS:
        feats = get_feature_names(engineered, fs)
        assert len(feats) >= 4, f"{fs} resolved to too few features"


def test_eng_with_vp_degrades_without_vp(engineered):
    """Uploads without VP must automatically drop Vp-dependent features."""
    no_vp = engineered.drop(columns=["VP", "PHIS", "AI"])
    feats = get_feature_names(no_vp, "eng_with_vp")
    assert "VP" not in feats and all("AI" not in f for f in feats)
    assert "GR" in feats


def test_clean_logs_drops_nan_rows(small_df):
    dirty = small_df.copy()
    dirty.loc[0, "GR"] = np.nan
    cleaned = clean_logs(dirty)
    assert not cleaned["GR"].isna().any()


def test_split_xy_core_targets(core_matrices):
    from geomech_logml.config import TARGETS
    X, Y, g = core_matrices
    assert list(Y.columns) == TARGETS
    assert len(X) == len(Y) == len(g)


# ---------------------------------------------------------------------------
# Well-wise CV
# ---------------------------------------------------------------------------
def test_wellkfold_no_well_leakage(core_matrices):
    X, Y, g = core_matrices
    splitter = WellKFold(n_splits=3, seed=1)
    seen_test_wells = []
    for tr, te in splitter.split(X, groups=g.to_numpy()):
        assert set(g.iloc[tr]) & set(g.iloc[te]) == set()
        seen_test_wells.extend(set(g.iloc[te]))
    # every well must be tested exactly once
    assert sorted(seen_test_wells) == sorted(g.unique())


def test_leave_one_well_out_covers_all(core_matrices):
    X, Y, g = core_matrices
    splitter = LeaveOneWellOut()
    n = sum(1 for _ in splitter.split(X, groups=g.to_numpy()))
    assert n == g.nunique()


def test_make_splitter_factory(core_matrices):
    _, _, g = core_matrices
    sp = make_splitter("well_kfold", n_wells=g.nunique(), n_splits=5)
    assert isinstance(sp, WellKFold)
    assert sp.get_n_splits(groups=g.to_numpy()) == min(5, g.nunique())
    assert isinstance(make_splitter("leave_one_well_out"), LeaveOneWellOut)
    with pytest.raises(ValueError):
        make_splitter("well_kfold", n_wells=1)  # cannot fold a single well
