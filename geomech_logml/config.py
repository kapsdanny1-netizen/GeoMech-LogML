"""Shared constants and column definitions for GeoMech-LogML.

A single source of truth for log/target names so that the data layer, feature
engineering, models, uncertainty and the Streamlit app never disagree.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED: int = 42

# ---------------------------------------------------------------------------
# Input wireline logs (standard / legacy suites only — NO shear sonic required)
# ---------------------------------------------------------------------------
BASE_LOGS: list[str] = ["GR", "RHOB", "NPHI", "RT"]
OPTIONAL_LOGS: list[str] = ["VP"]

#: Column aliases commonly found in LAS/CSV exports -> canonical name.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "DEPT": ("DEPT", "DEPTH", "MD", "DEPTHT", "TVD", "TVDSS", "DEPT_FS", "DEPTH_FS"),
    "GR": ("GR", "GRGC", "SGR", "ECGR", "CGR", "GR_EDTC"),
    "RHOB": ("RHOB", "RHOZ", "DEN", "RHOBL", "ZDEN", "RHOB_CORR"),
    "NPHI": ("NPHI", "NPHI_LS", "CNC", "NPOR", "TNPH", "NPHI_CORR"),
    "RT": ("RT", "RT90", "LLD", "ILD", "LL9", "AT90", "RDEP", "RILD", "HRLT"),
    "VP": ("VP", "VPLE", "VPLE_P", "SONVEL_VP", "VPVS"),  # VPVS last-resort? no: handled separately
}
# NOTE: VPVS is intentionally NOT an alias for VP (it is a ratio). Keep list strict;
# the alias table above is used only for exact-name mapping.

#: Sonic compressional transit time aliases (converted to VP = 1e6 / DTC_US_F, m/s)
DTC_ALIASES: tuple[str, ...] = ("DTC", "DT", "DT4P", "DTCO", "AC", "DT_RHOB")

# ---------------------------------------------------------------------------
# Prediction targets
# ---------------------------------------------------------------------------
TARGETS: list[str] = ["E_STAT", "NU_STAT", "UCS"]
TARGET_UNITS: dict[str, str] = {
    "E_STAT": "GPa",
    "NU_STAT": "dimensionless",
    "UCS": "MPa",
}
TARGET_LABELS: dict[str, str] = {
    "E_STAT": "Static Young's modulus",
    "NU_STAT": "Static Poisson's ratio",
    "UCS": "Unconfined compressive strength",
}
#: Physically admissible target ranges (used for clipping predictions + QC plots).
TARGET_BOUNDS: dict[str, tuple[float, float]] = {
    "E_STAT": (0.1, 60.0),
    "NU_STAT": (0.05, 0.49),
    "UCS": (0.1, 150.0),
}

# ---------------------------------------------------------------------------
# Ground-truth / QC columns produced by the synthetic generator.
# NEVER used as model features — they exist so notebooks/tests/QC can verify physics.
# ---------------------------------------------------------------------------
TRUTH_COLUMNS: list[str] = [
    "FACIES",
    "VSH_TRUE",
    "PHI_TRUE",
    "PP_MPA",
    "SV_MPA",
    "SIG_EFF_MPA",
    "E_DYN",
    "NU_DYN",
    "IS_HC",
]

#: Marker column indicating lab-measured (paired static-dynamic "core") rows.
CORE_FLAG: str = "IS_CORE"

#: Well identifier / depth columns
WELL_COL: str = "WELL"
DEPTH_COL: str = "DEPT"

# ---------------------------------------------------------------------------
# Facies codes used by the synthetic generator (also used for plot colours)
# ---------------------------------------------------------------------------
FACIES_NAMES: dict[int, str] = {
    0: "Sand",
    1: "Shaly sand",
    2: "Sandy shale",
    3: "Shale",
}
