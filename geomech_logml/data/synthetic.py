"""Physics-based synthetic multi-well log generator for the Agbada Formation (Niger Delta).

This generator produces *log-like* data whose petrophysics is consistent with published
ranges for unconsolidated, overpressured, mixed sand–shale deltaic sequences
(the Agbada Formation of the Niger Delta). It exists so that the whole ML system can be
developed, tested and demonstrated without proprietary data — while remaining fully
compatible with real LAS/CSV files supplied by the user.

Forward model (per well, top-down; every step is vectorised over depth)
-----------------------------------------------------------------------
1. **Facies succession** — a 4-state Markov chain (Sand / Shaly sand / Sandy shale /
   Shale) with log-normal bed thickness. Channel sands and thick shales dominate,
   mimicking an aggradational deltaic parasequence stack.
2. **Clay volume (Vsh_true)** — facies-controlled, ~N(clay_mean, 0.05), clipped to [0, 0.98].
3. **Pore pressure** — hydrostatic above a per-well overpressure top `z_op`
   (Agbada overpressure onset is commonly reported around 2.4–3.1 km);
   below it, pore pressure ramps toward a fraction (lambda_max ~ 0.75–0.85) of the
   lithostatic stress (undercompaction mechanism; cf. Hottman & Johnson, 1965).
4. **Effective stress** sigma_eff = Sv − Pp (Terzaghi).
5. **Porosity** — Athy-style exponential compaction driven by *effective* stress
   (so overpressure preserves porosity), with facies-dependent surface porosity/
   decay constants and a mild quartz-cementation penalty for sands > 3.5 km.
6. **Wireline logs** (the ONLY model inputs):
   * GR — linear mixing of clean-sand (~22 API) and shale (~105 API) endmembers.
   * RHOB — matrix/fluid volume mixing (quartz 2.65, clay ~2.68, brine ~1.02 g/cc;
     hydrocarbon zones diluted to ~0.75 g/cc).
   * NPHI — hydrogen-index mixing + shale bound-water term + sandstone lattice offset.
   * RT — parallel shale/Archie conductivity mixing, Archie (1942) with
     m = 2.0 − 0.5·Vsh, n = 2, hydrocarbon sands with Sw ∈ [0.15, 0.65].
   * VP — Han et al. (1986) brine-sand velocities with unconsolidated-sediment
     pressure sensitivity  Vp ∝ (sigma_eff/sigma_ref)^beta  (gas sands damped ~12%).
7. **Geomechanics** — Vs (HIDDEN: never a model input) from the Han shale-sand line;
   dynamic moduli Edyn, nu_dyn from Vp/Vs/rho; then the *static* properties:
   * E_STAT  = Edyn · f, with the static/dynamic ratio f a *learnable* function of
     rock quality:  f = 0.72 − 0.42·(Edyn/40)  (weak, porous rocks show a large
     static–dynamic mismatch; competent rocks approach 1; cf. Zoback, 2007; Plumb, 1994)
     plus per-well and depth-correlated (core-plug-scale) noise.
   * NU_STAT  ≈ 0.96 · nu_dyn + small offset/noise, clipped to [0.08, 0.42].
   * UCS — porosity- and clay-controlled empirical strength (cf. Chang et al., 2006;
     Lashkaripour & Dusseault, 1993 for weak shaly sands):
     sand 130·exp(−11·phi) MPa, shale 38·exp(−6.2·phi) MPa, blended by Vsh^0.8,
     scaled by an effective-stress (compaction-strength) factor and per-well/plug noise.
8. **Core plugs** — a small fraction of depths flagged `IS_CORE=1` (clustered in
   reservoir-quality sands with occasional shale plugs). These are the *paired
   static measurements* the models train on — exactly as in a real core-calibration
   workflow. `IS_CORE=0` rows carry ground-truth properties for evaluation only.

Everything is deterministic given `seed`. All generated values respect published
Agbada property ranges (documented per-parameter in `SyntheticConfig`).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from geomech_logml.config import CORE_FLAG, DEPTH_COL, WELL_COL

__all__ = ["SyntheticConfig", "generate_well", "generate_dataset", "config_to_dict"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    """Controls for the Agbada-like synthetic generator.

    All ranges are in SI-ish oilfield units: depth in metres, moduli in GPa,
    pressures in MPa, VP in m/s, resistivity in ohm·m.
    """

    # --- wells & sampling --------------------------------------------------
    n_wells: int = 8
    depth_min_m: float = 900.0
    depth_max_m: float = 4200.0
    step_m: float = 0.5
    well_prefix: str = "AGB"

    # --- overpressure --------------------------------------------------------
    overpressure: bool = True
    overpressure_top_range_m: tuple[float, float] = (2400.0, 3100.0)
    lambda_max_range: tuple[float, float] = (0.72, 0.82)  # Pp/Sv at TD if overpressured
    overpressure_exponent: float = 1.6  # ramp curvature below top

    # --- stress gradients (MPa/m) --------------------------------------------
    sv_gradient: float = 0.0231          # ~2.35 g/cc overburden
    hydro_gradient: float = 0.0101       # ~normal pressure, brine
    gradient_jitter: float = 0.04        # per-well relative variation of gradients

    # --- petrophysical endmembers ---------------------------------------------
    gr_sand: float = 22.0                # API
    gr_shale: float = 105.0              # API
    gr_noise: float = 3.0
    rho_matrix_sand: float = 2.65        # g/cc quartz
    rho_matrix_clay: float = 2.68        # g/cc dry clay
    rho_brine: float = 1.02
    rho_hc: float = 0.75
    rhob_noise: float = 0.015
    nphi_shale_term: float = 0.14        # bound-water hydrogen index contribution
    nphi_matrix_offset: float = -0.02
    nphi_noise: float = 0.012
    rw_range: tuple[float, float] = (0.02, 0.06)   # formation-water resistivity (ohm·m)
    rt_noise: float = 0.28               # log-normal multiplicative noise

    # --- porosity (Athy compaction vs effective stress) ------------------------
    phi0_sand: float = 0.42
    phi0_shale: float = 0.53
    k_sand: float = 0.023                # 1/MPa decay constant
    k_shale: float = 0.016               # 1/MPa (undercompacted shales retain porosity)
    phi_noise: float = 0.045             # log-normal multiplicative
    cementation_depth_m: float = 3500.0  # quartz overgrowth onset for sands
    cementation_rate: float = 7.0e-5     # porosity fraction lost per metre below onset

    # --- velocities (Han et al. 1986 + pressure scaling) ------------------------
    han_vp: tuple[float, float, float] = (5.59, 6.93, 2.18)   # Vp = a - b*phi - c*Vsh (km/s)
    han_vs: tuple[float, float, float] = (3.52, 4.91, 1.89)   # Vs = a - b*phi - c*Vsh (km/s)
    sigma_ref_mpa: float = 40.0          # Han's effective-stress reference
    beta_vp: float = 0.145               # unconsolidated pressure sensitivity exponent
    beta_vs: float = 0.120
    vp_gas_damp: float = 0.88            # Gassmann-ish brine->gas reduction in pay
    vp_noise: float = 0.075              # km/s Gaussian
    vp_floor_kms: float = 1.45
    vs_floor_kms: float = 0.55

    # --- static-dynamic conversion (LEARNED BY THE MODEL, never hard-coded) ------
    sd_ratio_high: float = 0.72          # E_stat/E_dyn at Edyn -> 0
    sd_ratio_span: float = 0.42          # linear decrease over Edyn in [0, 40] GPa
    sd_well_sigma: float = 0.035         # per-well geologic factor
    sd_plug_sigma: float = 0.025         # depth-correlated (plug-scale) noise, gaussian-smoothed
    sd_smooth_samples: int = 25          # depth correlation length for plug-scale noise
    nu_static_scale: float = 0.96
    nu_static_offset: float = -0.015
    nu_noise: float = 0.010

    # --- strength ---------------------------------------------------------------
    ucs_sand_A: float = 130.0            # MPa
    ucs_sand_B: float = 11.0             # UCS = A*exp(-B*phi)
    ucs_shale_A: float = 26.0            # weak undercompacted shales (Lashkaripour & Dusseault, 1993)
    ucs_shale_B: float = 6.4
    ucs_stress_ref_mpa: float = 35.0     # compaction-strength factor saturates here
    ucs_well_sigma: float = 0.13
    ucs_plug_sigma: float = 0.10

    # --- hydrocarbons ---------------------------------------------------------
    hc_probability_per_well: float = 0.6  # chance a well penetrates at least one pay sand
    sw_range: tuple[float, float] = (0.15, 0.65)

    # --- core plugs -------------------------------------------------------------
    core_rate_sand: float = 0.045        # P(plug) per 0.5 m depth increment in sand
    core_rate_shaly_sand: float = 0.020
    core_rate_shale: float = 0.006
    core_min_gap_m: float = 2.0          # minimum spacing between plugs

    seed: int = 42

    # ------------------------------------------------------------------
    def well_ids(self) -> list[str]:
        return [f"{self.well_prefix}-{i + 1:02d}" for i in range(self.n_wells)]


def config_to_dict(cfg: SyntheticConfig) -> dict:
    """Serialise a config (dataclass -> plain dict, JSON-safe)."""
    return asdict(cfg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_FACIES_STATES = ("sand", "shaly_sand", "sandy_shale", "shale")

#: Mean clay volume per facies state
_FACIES_VSH = {"sand": 0.08, "shaly_sand": 0.25, "sandy_shale": 0.55, "shale": 0.86}
_FACIES_VSH_SD = 0.045

#: Mean bed thickness (m), log-normal
_FACIES_THICK = {"sand": 16.0, "shaly_sand": 10.0, "sandy_shale": 14.0, "shale": 38.0}

#: Markov transition matrix (rows = from, cols = to). Strong shale->sand->shale
#: channel stacking; direct sand->sand promotes thick blocky channels.
_TRANSITION = np.array(
    [
        # to:  sand  shaly  sandy  shale
        [0.35, 0.30, 0.15, 0.20],  # from sand
        [0.25, 0.20, 0.25, 0.30],  # from shaly sand
        [0.15, 0.25, 0.25, 0.35],  # from sandy shale
        [0.30, 0.20, 0.20, 0.30],  # from shale
    ]
)

_FACIES_CODE = {name: code for code, name in enumerate(
    ["Sand", "Shaly sand", "Sandy shale", "Shale"])}


def _gaussian_smooth(x: np.ndarray, sigma: int) -> np.ndarray:
    """Depth-correlated noise via a gaussian kernel (reflect-padded 1-D convolution)."""
    if sigma <= 0 or x.size == 0:
        return x
    kernel = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, 3 * sigma, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _markov_facies(rng: np.random.Generator, n: int, step: float) -> np.ndarray:
    """Simulate a facies profile as a sequence of Markov-switched beds.

    Returns an integer array of facies codes (0=sand..3=shale), length n.
    """
    facies = np.empty(n, dtype=np.int8)
    pos = 0
    state = "shale"  # deltaic wells typically start in the shale-prone top Akata/Agbada
    while pos < n:
        nxt_idx = rng.choice(4, p=_TRANSITION[_FACIES_STATES.index(state)])
        state = _FACIES_STATES[nxt_idx]
        mean_thick = _FACIES_THICK[state]
        thick = int(max(1, rng.lognormal(np.log(mean_thick), 0.55) / step))
        facies[pos: pos + thick] = _FACIES_CODE[
            {"sand": "Sand", "shaly_sand": "Shaly sand",
             "sandy_shale": "Sandy shale", "shale": "Shale"}[state]
        ]
        pos += thick
    return facies


# ---------------------------------------------------------------------------
# Per-well forward model
# ---------------------------------------------------------------------------
def generate_well(well: str, cfg: SyntheticConfig, seed_offset: int = 0) -> pd.DataFrame:
    """Generate one synthetic well log suite with known ground-truth geomechanics.

    Parameters
    ----------
    well : str
        Well identifier written to the `WELL` column.
    cfg : SyntheticConfig
        Generator configuration (ranges/coefficients).
    seed_offset : int
        Offset added to `cfg.seed` so each well gets an independent but
        deterministic random stream.

    Returns
    -------
    pd.DataFrame
        Depth-indexed frame with input logs (GR, RHOB, NPHI, RT, VP), targets
        (E_STAT, NU_STAT, UCS), truth/QC columns and the IS_CORE flag.
    """
    rng = np.random.default_rng(cfg.seed + seed_offset * 1000 + 17)

    n = int(np.floor((cfg.depth_max_m - cfg.depth_min_m) / cfg.step_m)) + 1
    depth = cfg.depth_min_m + cfg.step_m * np.arange(n)

    # 1. Facies & clay volume --------------------------------------------------
    facies = _markov_facies(rng, n, cfg.step_m)
    vsh_mean = np.array([_FACIES_VSH[_FACIES_STATES[f]] for f in facies])
    vsh = np.clip(vsh_mean + rng.normal(0, _FACIES_VSH_SD, n), 0.02, 0.98)

    # 2. Pressure & stress -------------------------------------------------------
    grad_scale = 1.0 + rng.normal(0, cfg.gradient_jitter)
    sv = cfg.sv_gradient * grad_scale * depth                      # lithostatic, MPa
    hydro = cfg.hydro_gradient * (1.0 + rng.normal(0, 0.3 * cfg.gradient_jitter)) * depth
    if cfg.overpressure:
        z_op = rng.uniform(*cfg.overpressure_top_range_m)
        lam_max = rng.uniform(*cfg.lambda_max_range)
        ramp = np.clip((depth - z_op) / max(cfg.depth_max_m - z_op, 1.0), 0.0, 1.0) ** cfg.overpressure_exponent
        pp = hydro + (lam_max * sv - hydro) * ramp
    else:
        z_op = np.inf
        pp = hydro.copy()
    sig_eff = np.clip(sv - pp, 1.5, None)                          # Terzaghi effective stress, MPa

    # 3. Porosity (Athy vs effective stress + deep sand cementation) ----------------
    phi0 = cfg.phi0_sand + (cfg.phi0_shale - cfg.phi0_sand) * vsh
    k_decay = cfg.k_sand + (cfg.k_shale - cfg.k_sand) * vsh
    phi = phi0 * np.exp(-k_decay * sig_eff)
    phi -= cfg.cementation_rate * np.clip(depth - cfg.cementation_depth_m, 0, None) * (1.0 - vsh)
    phi = np.clip(phi * np.exp(rng.normal(0, cfg.phi_noise, n)), 0.015, 0.55)

    # 4. Hydrocarbon pay (per-well; brine default) ---------------------------------
    is_hc = np.zeros(n, dtype=bool)
    if rng.random() < cfg.hc_probability_per_well:
        sand_idx = np.where((facies <= 1) & (phi > 0.18))[0]
        if sand_idx.size > 40:
            start = int(rng.integers(0, sand_idx.size - 30))
            column_len = int(rng.integers(20, max(21, min(80, sand_idx.size - start))))
            is_hc[sand_idx[start: start + column_len]] = True
    sw = np.where(is_hc, rng.uniform(*cfg.sw_range), 1.0)

    # 5. Wireline logs (the ONLY permitted model inputs) ----------------------------
    gr = np.clip(cfg.gr_sand + (cfg.gr_shale - cfg.gr_sand) * vsh ** 0.9
                 + rng.normal(0, cfg.gr_noise, n), 8, 180)

    rho_fluid = np.where(is_hc, cfg.rho_hc, cfg.rho_brine)
    rho_matrix = cfg.rho_matrix_sand + (cfg.rho_matrix_clay - cfg.rho_matrix_sand) * vsh
    rhob = phi * rho_fluid + (1.0 - phi) * rho_matrix + rng.normal(0, cfg.rhob_noise, n)
    rhob = np.clip(rhob, 1.45, 2.95)

    hi_fluid = np.where(is_hc, 0.65, 1.0)   # neutron response to hydrocarbon
    nphi = phi * hi_fluid + (1.0 - phi) * vsh * cfg.nphi_shale_term \
        + cfg.nphi_matrix_offset * (1.0 - vsh) + rng.normal(0, cfg.nphi_noise, n)
    nphi = np.clip(nphi, 0.0, 0.50)

    # Resistivity: parallel shale / Archie conductivities. The shale branch is
    # weighted by vsh^1.2 so clean sands keep their Archie response (a sand with
    # 8% clay is not short-circuited by shale conduction).
    rw = rng.uniform(*cfg.rw_range)
    m_cem = np.clip(2.0 - 0.5 * vsh, 1.4, 2.1)
    f_formation = phi ** (-m_cem)                          # formation factor
    rt_sand = f_formation * rw / np.clip(sw, 0.08, 1.0) ** 2  # Archie, n=2
    rt_shale = 1.1 * (depth / 1000.0) ** 1.15 + 0.4        # compaction trend, ohm·m
    w_shale = vsh ** 1.2
    cond = w_shale / np.clip(rt_shale, 0.2, None) + (1.0 - w_shale) / np.clip(rt_sand, 0.05, None)
    rt = np.clip(1.0 / np.clip(cond, 1e-4, None) * np.exp(rng.normal(0, cfg.rt_noise, n)),
                 0.3, 3000.0)

    # 5b. Compressional velocity (Han + pressure sensitivity) — Vs stays HIDDEN -----
    a_v, b_v, c_v = cfg.han_vp
    vp_kms = (a_v - b_v * phi - c_v * vsh) * (sig_eff / cfg.sigma_ref_mpa) ** cfg.beta_vp
    vp_kms = np.where(is_hc, vp_kms * cfg.vp_gas_damp, vp_kms)
    vp_kms = np.clip(vp_kms + rng.normal(0, cfg.vp_noise, n), cfg.vp_floor_kms, 5.6)
    vp = vp_kms * 1000.0                                   # m/s

    a_s, b_s, c_s = cfg.han_vs
    vs_kms = (a_s - b_s * phi - c_s * vsh) * (sig_eff / cfg.sigma_ref_mpa) ** cfg.beta_vs
    vs_kms = np.where(is_hc, vs_kms * 0.95, vs_kms)
    vs_kms = np.clip(vs_kms, cfg.vs_floor_kms, 3.4)

    # 6. Dynamic moduli (need Vs — available only inside the generator) ---------------
    rho_gcc = rhob
    vp2, vs2 = vp_kms ** 2, vs_kms ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        e_dyn = rho_gcc * vs2 * (3 * vp2 - 4 * vs2) / np.clip(vp2 - vs2, 1e-6, None)
        nu_dyn = (vp2 - 2 * vs2) / np.clip(2 * (vp2 - vs2), 1e-6, None)
    e_dyn = np.clip(e_dyn, 0.3, 45.0)
    nu_dyn = np.clip(nu_dyn, 0.10, 0.45)

    # 7. Static properties (ground truth) — mismatch is a LEARNABLE function of rock
    #    quality; the models must recover it from logs alone (no hard-coded factor).
    well_sd_factor = rng.normal(0, cfg.sd_well_sigma)
    plug_noise = _gaussian_smooth(rng.normal(0, cfg.sd_plug_sigma, n), cfg.sd_smooth_samples)
    sd_ratio = np.clip(cfg.sd_ratio_high - cfg.sd_ratio_span * (e_dyn / 40.0)
                       + well_sd_factor + plug_noise, 0.22, 0.88)
    e_stat = np.clip(e_dyn * sd_ratio, 0.4, 40.0)
    nu_stat = np.clip(cfg.nu_static_scale * nu_dyn + cfg.nu_static_offset
                      + rng.normal(0, cfg.nu_noise, n), 0.08, 0.42)

    ucs_sand = cfg.ucs_sand_A * np.exp(-cfg.ucs_sand_B * phi)
    ucs_shale = cfg.ucs_shale_A * np.exp(-cfg.ucs_shale_B * phi)
    stress_factor = 0.55 + 0.45 * np.clip(sig_eff / cfg.ucs_stress_ref_mpa, 0.0, 1.0)
    wfac = np.exp(rng.normal(0, cfg.ucs_well_sigma))
    ucs = (ucs_sand * (1.0 - vsh ** 0.8) + ucs_shale * vsh ** 0.8) * stress_factor * wfac
    ucs = np.clip(ucs * np.exp(rng.normal(0, cfg.ucs_plug_sigma, n)), 0.5, 95.0)

    # 8. Core plugs — the paired static measurements used for training ---------------
    # Rates are defined per 0.5 m of rock; convert to per-row probability so the
    # plug intensity per metre is invariant to the sampling step.
    rate = np.select(
        [facies == 0, facies == 1, facies == 2],
        [cfg.core_rate_sand, cfg.core_rate_shaly_sand, cfg.core_rate_shale],
        default=cfg.core_rate_shale * 0.7,
    ) * (cfg.step_m / 0.5)                                   # per-ROW probability
    draws = rng.random(n) < rate
    is_core = draws.copy()
    # enforce minimum plug spacing (keep the shallowest of any cluster)
    min_gap = int(cfg.core_min_gap_m / cfg.step_m)
    last = -10**9
    for i in np.where(draws)[0]:
        if i - last < min_gap:
            is_core[i] = False
        else:
            last = i

    df = pd.DataFrame(
        {
            WELL_COL: well,
            DEPTH_COL: np.round(depth, 2),
            "GR": np.round(gr, 2),
            "RHOB": np.round(rhob, 4),
            "NPHI": np.round(nphi, 4),
            "RT": np.round(rt, 3),
            "VP": np.round(vp, 1),
            "E_STAT": np.round(e_stat, 4),
            "NU_STAT": np.round(nu_stat, 4),
            "UCS": np.round(ucs, 2),
            CORE_FLAG: is_core.astype(np.int8),
            "FACIES": facies.astype(np.int8),
            "VSH_TRUE": np.round(vsh, 4),
            "PHI_TRUE": np.round(phi, 4),
            "PP_MPA": np.round(pp, 3),
            "SV_MPA": np.round(sv, 3),
            "SIG_EFF_MPA": np.round(sig_eff, 3),
            "E_DYN": np.round(e_dyn, 4),
            "NU_DYN": np.round(nu_dyn, 4),
            "IS_HC": is_hc.astype(np.int8),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Dataset-level API
# ---------------------------------------------------------------------------
def generate_dataset(cfg: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate a full multi-well dataset (default: 8 Agbada-like wells, seed=42).

    Returns a single DataFrame with all wells stacked and a `WELL` column.
    """
    cfg = cfg or SyntheticConfig()
    frames = [generate_well(w, cfg, seed_offset=i) for i, w in enumerate(cfg.well_ids())]
    return pd.concat(frames, ignore_index=True)


def save_dataset(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Save a generated dataset as one CSV per well plus a combined CSV.

    LAS writing is handled by :mod:`geomech_logml.data.las_io` (used by the app/CLI).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for well, g in df.groupby(WELL_COL):
        p = out / f"{well}.csv"
        g.to_csv(p, index=False)
        paths.append(p)
    combined = out / "all_wells.csv"
    df.to_csv(combined, index=False)
    paths.append(combined)
    return paths
