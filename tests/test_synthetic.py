"""Synthetic generator: physical realism, published-range compliance, determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomech_logml.config import CORE_FLAG, TARGETS, WELL_COL
from geomech_logml.data.synthetic import generate_well

def test_generator_is_deterministic(small_cfg):
    a = generate_well("T-1", small_cfg, seed_offset=0)
    b = generate_well("T-1", small_cfg, seed_offset=0)
    pd.testing.assert_frame_equal(a, b)


def test_different_wells_differ(small_cfg):
    a = generate_well("T-1", small_cfg, seed_offset=0)
    b = generate_well("T-2", small_cfg, seed_offset=1)
    assert not np.allclose(a["GR"].to_numpy(), b["GR"].to_numpy())


def test_all_required_columns_present(small_df):
    required = ["DEPT", "GR", "RHOB", "NPHI", "RT", "VP"] + TARGETS + [CORE_FLAG, WELL_COL]
    for col in required:
        assert col in small_df.columns, f"missing {col}"


def test_property_ranges_respect_agbada_context(small_df):
    """Ranges typical of unconsolidated, overpressured siliciclastics."""
    bounds = {
        "GR": (10, 160), "RHOB": (1.7, 2.75), "NPHI": (0.0, 0.5),
        "RT": (0.3, 3000), "VP": (1400, 5500),
        "E_STAT": (0.4, 30), "NU_STAT": (0.08, 0.42), "UCS": (0.5, 80),
    }
    for col, (lo, hi) in bounds.items():
        assert small_df[col].min() >= lo, f"{col} below physical bound"
        assert small_df[col].max() <= hi, f"{col} above physical bound"


def test_compaction_driven_by_effective_stress(small_df):
    """Porosity must compact with EFFECTIVE stress (Athy) — the mechanism by which
    overpressure preserves porosity. Checked within facies to avoid lithology mixing."""
    from scipy.stats import spearmanr
    for facies_code in (0, 3):  # sand and shale
        sub = small_df[small_df["FACIES"] == facies_code]
        rho, _ = spearmanr(sub["PHI_TRUE"], sub["SIG_EFF_MPA"])
        assert rho < -0.3, f"facies {facies_code}: phi vs sigma_eff correlation {rho:.2f}"


def test_overpressure_reduces_effective_stress(small_df):
    """Where present, the ramp must drive pore pressure above hydrostatic."""
    d = small_df.assign(hydro=0.0101 * small_df["DEPT"])
    excess = (d["PP_MPA"] - d["hydro"])
    assert excess.max() > 3.0, "no well developed meaningful overpressure"
    assert (excess < 0.5).mean() > 0.2, "upper section should be normally pressured"


def test_clay_control_on_strength(small_df):
    """At comparable porosity, cleaner sand must be stronger than clay-rich rock."""
    band = small_df[small_df["PHI_TRUE"].between(0.20, 0.28)]
    sand = band.loc[band["VSH_TRUE"] < 0.2, "UCS"].mean()
    shale = band.loc[band["VSH_TRUE"] > 0.7, "UCS"].mean()
    assert sand > shale


def test_static_dynamic_ratio_is_variable_and_learnable(small_df):
    """Requirement: no hard-coded factor — the ratio must vary with rock quality."""
    ratio = small_df["E_STAT"] / small_df["E_DYN"]
    assert ratio.std() > 0.03, "static/dynamic ratio looks constant"
    # stronger (higher-Edyn) rocks should have a LARGER mismatch gap
    low_e = small_df.loc[small_df["E_DYN"] < 10, "E_STAT"] / small_df.loc[small_df["E_DYN"] < 10, "E_DYN"]
    high_e = small_df.loc[small_df["E_DYN"] > 25, "E_STAT"] / small_df.loc[small_df["E_DYN"] > 25, "E_DYN"]
    assert low_e.mean() > high_e.mean()


def test_no_shear_sonic_in_inputs(small_df):
    """Vs must NOT exist as a deliverable log column (hidden inside generator only)."""
    assert "VS" not in small_df.columns
    assert "DTSM" not in small_df.columns


def test_core_plugs_are_a_small_subset(small_df):
    frac = small_df[CORE_FLAG].mean()
    assert 0.002 < frac < 0.15, f"core fraction {frac:.4f} outside plausible range"
    assert small_df.groupby(WELL_COL)[CORE_FLAG].sum().min() >= 20, "each well needs plugs"
