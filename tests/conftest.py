"""Shared fixtures: one small synthetic dataset per session (fast tests)."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from geomech_logml.data.synthetic import SyntheticConfig, generate_dataset
from geomech_logml.preprocessing.features import clean_logs, engineer_features, split_xy_core


@pytest.fixture(scope="session")
def small_cfg() -> SyntheticConfig:
    return SyntheticConfig(
        n_wells=5, depth_min_m=1000.0, depth_max_m=4000.0, step_m=2.0, seed=123,
    )


@pytest.fixture(scope="session")
def small_df(small_cfg) -> object:
    """Raw synthetic frame (5 wells, 2 m step)."""
    return generate_dataset(small_cfg)


@pytest.fixture(scope="session")
def engineered(small_df) -> object:
    return engineer_features(clean_logs(small_df))


@pytest.fixture(scope="session")
def core_matrices(engineered) -> tuple:
    X, Y, g = split_xy_core(engineered, "eng_with_vp")
    assert len(X) > 50, "expected a usable number of core plugs"
    return X, Y, g
