"""GeoMech-LogML: ML prediction of rock strength & elastic properties from standard wireline logs.

Designed for unconsolidated siliciclastic settings (Niger Delta Agbada Formation).
Inputs: GR, RHOB, NPHI, Rt (+ optionally Vp). Targets: E_static, Poisson's ratio, UCS.
"""

from geomech_logml.config import (
    BASE_LOGS,
    OPTIONAL_LOGS,
    TARGETS,
    TARGET_UNITS,
    GLOBAL_SEED,
)

__version__ = "0.3.0"

__all__ = [
    "BASE_LOGS",
    "OPTIONAL_LOGS",
    "TARGETS",
    "TARGET_UNITS",
    "GLOBAL_SEED",
    "__version__",
]
