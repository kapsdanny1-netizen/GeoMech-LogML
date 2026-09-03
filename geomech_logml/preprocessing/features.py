"""Log cleaning and petrophysical feature engineering.

Engineered features (all derivable at the wellsite from standard logs):

* ``RT_LOG10`` — log10 resistivity (resistivity is log-distributed).
* ``VSH_GR``   — shale volume from GR via the Larionov (1969) *young-rocks* equation,
                 normalised per well with robust (P2/P98) endmembers.
* ``PHID``     — density porosity (matrix 2.65 g/cc, fluid 1.0 g/cc).
* ``PHIND``    — neutron–density average porosity (gas-resilient).
* ``PHIS``     — Wyllie sonic porosity from VP (only when VP exists).
* ``AI``       — acoustic impedance RHOB·VP (only when VP exists).
* ``VPHI_RATIO``— NPHI/PHID ratio (gas/shale indicator).

Feature sets for the with/without-Vp ablation are defined in :data:`FEATURE_SETS`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomech_logml.config import (
    BASE_LOGS,
    CORE_FLAG,
    DEPTH_COL,
    TARGETS,
    WELL_COL,
)

__all__ = [
    "clean_logs",
    "engineer_features",
    "FEATURE_SETS",
    "get_feature_names",
    "build_feature_matrix",
]

#: Physically admissible log ranges (rows outside are clipped; NaNs dropped).
LOG_BOUNDS: dict[str, tuple[float, float]] = {
    "GR": (0.0, 250.0),      # API
    "RHOB": (1.4, 3.05),     # g/cc
    "NPHI": (-0.05, 0.6),    # v/v (small negative readings are real in gas sands)
    "RT": (0.1, 5000.0),     # ohm·m
    "VP": (1200.0, 6500.0),  # m/s
    DEPTH_COL: (0.0, 12000.0),  # m
}

#: Raw + engineered feature sets (ablation axis: with vs without VP).
FEATURE_SETS: dict[str, dict] = {
    "raw_no_vp": {
        "label": "Raw logs — no Vp (legacy triple-combo only)",
        "requires_vp": False,
        "features": ["GR", "RHOB", "NPHI", "RT_LOG10"],
    },
    "raw_with_vp": {
        "label": "Raw logs — with Vp",
        "requires_vp": True,
        "features": ["GR", "RHOB", "NPHI", "RT_LOG10", "VP"],
    },
    "eng_no_vp": {
        "label": "Engineered — no Vp (legacy suite + petrophysics)",
        "requires_vp": False,
        "features": ["DEPT", "GR", "RHOB", "NPHI", "RT_LOG10", "VSH_GR", "PHID", "PHIND"],
    },
    "eng_with_vp": {
        "label": "Engineered — with Vp (recommended)",
        "requires_vp": True,
        "features": [
            "DEPT", "GR", "RHOB", "NPHI", "RT_LOG10", "VSH_GR", "PHID", "PHIND",
            "VP", "PHIS", "AI", "VPHI_RATIO",
        ],
    },
}
DEFAULT_FEATURE_SET = "eng_with_vp"


def clean_logs(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Clip logs to physical ranges and drop rows with missing essential curves.

    Parameters
    ----------
    df : canonical log frame (WELL, DEPT, GR, RHOB, NPHI, RT, [VP], ...)
    strict : if True, require all four base logs (GR, RHOB, NPHI, RT). Rows with
        NaN in *optional* logs (VP) are kept — the no-Vp feature set remains usable.

    Returns
    -------
    cleaned copy (rows reordered never; index reset).
    """
    out = df.copy()
    for col, (lo, hi) in LOG_BOUNDS.items():
        if col in out.columns:
            out[col] = out[col].clip(lo, hi)

    essential = BASE_LOGS + [WELL_COL, DEPTH_COL] if strict else [WELL_COL, DEPTH_COL]
    out = out.dropna(subset=[c for c in essential if c in out.columns]).reset_index(drop=True)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered petrophysical features (vectorised; per-well GR normalisation).

    Never mutates the input; returns a copy. Features requiring VP are NaN where
    VP is missing, so ``build_feature_matrix`` can exclude them cleanly.
    """
    out = df.copy()
    out["RT_LOG10"] = np.log10(out["RT"].clip(lower=0.1))

    # --- Vshale (Larionov young-rocks), robust per-well GR endmembers -----------
    gr_min = out.groupby(WELL_COL)["GR"].transform(lambda s: np.nanpercentile(s, 2))
    gr_max = out.groupby(WELL_COL)["GR"].transform(lambda s: np.nanpercentile(s, 98))
    igr = ((out["GR"] - gr_min) / (gr_max - gr_min).clip(lower=1.0)).clip(0, 1)
    out["VSH_GR"] = np.clip(0.083 * (2.0 ** (3.7 * igr) - 1.0), 0.0, 1.0)

    # --- Porosity family ----------------------------------------------------------
    out["PHID"] = ((2.65 - out["RHOB"]) / (2.65 - 1.0)).clip(-0.05, 0.6)
    out["PHIND"] = ((out["PHID"] + out["NPHI"]) / 2.0).clip(-0.05, 0.6)
    out["VPHI_RATIO"] = (out["NPHI"] / out["PHID"].clip(lower=0.02)).clip(0.2, 10.0)
    if "VP" in out.columns:
        dtc = 1e6 / out["VP"].clip(lower=1200.0)          # us/ft
        out["PHIS"] = ((dtc - 55.5) / (189.0 - 55.5)).clip(0.0, 0.6)
        out["AI"] = out["RHOB"] * out["VP"] / 1000.0      # (g/cc)·(km/s)
    return out


def get_feature_names(df: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Resolve the ordered feature list for a named feature set, restricted to the
    columns actually present in ``df`` (so no-Vp data automatically degrade)."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set '{feature_set}'. Choose from {list(FEATURE_SETS)}")
    feats = FEATURE_SETS[feature_set]["features"]
    return [f for f in feats if f in df.columns]


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the (X, feature_names) matrix for modelling.

    The frame must already be cleaned (:func:`clean_logs`) and feature-engineered
    (:func:`engineer_features`); rows with NaN among the chosen features are dropped.
    """
    feats = get_feature_names(df, feature_set)
    if not feats:
        raise ValueError(f"Feature set '{feature_set}' produced no usable columns — "
                         f"check that the input logs exist.")
    sub = df.dropna(subset=feats).reset_index(drop=True)
    return sub[feats].astype(float), feats


def split_xy_core(
    df: pd.DataFrame,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (X, Y, groups) restricted to core-calibrated (paired) rows.

    This is the training view: models learn the static-dynamic relationship from
    rows where static lab measurements exist (`IS_CORE == 1`).
    """
    feats = get_feature_names(df, feature_set)
    needed = feats + TARGETS + [WELL_COL]
    core = df[df[CORE_FLAG] == 1].dropna(subset=[c for c in needed if c in df.columns])
    core = core.dropna(subset=feats)  # cannot use rows with missing features
    return core[feats].astype(float), core[TARGETS].astype(float), core[WELL_COL]
