# ⛰️ GeoMech-LogML

**Machine-learning prediction of rock strength and elastic properties from standard wireline logs — built for unconsolidated siliciclastic settings (Niger Delta · Agbada Formation).**

GeoMech-LogML predicts **static Young's modulus (E_static, GPa)**, **static Poisson's ratio (ν)** and **UCS (MPa)** from the legacy logging suite everyone already has — **GR, RHOB, NPHI, Rt**, plus **optional compressional velocity (Vp)** — with strict well-wise validation, calibrated uncertainty bands and full SHAP interpretability. No shear-wave / dipole sonic is required anywhere in the workflow.

| | |
|---|---|
| **Models** | Random Forest · XGBoost · shallow MLP (compared side-by-side) |
| **Validation** | K-fold / leave-one-well-out on *whole wells* — never random row splits |
| **Uncertainty** | Split-conformal prediction (well-wise calibrated) + Quantile Regression Forests |
| **Explainability** | SHAP beeswarm, dependence plots and per-depth waterfalls for every model |
| **Interface** | Streamlit dashboard (upload LAS/CSV or generate Agbada-like synthetic data) |
| **Export** | Curve predictions + metrics (CSV), report (Markdown **and multi-page PDF with charts**) |
| **Reproducibility** | Fixed seeds, pinned requirements, Dockerfile, pytest suite (39 tests), example notebooks |

---

## 1. Quick start

```bash
# 1) clone and enter
cd GeoMech-LogML

# 2) install (Python 3.11+)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                    # makes `geomech_logml` importable

# 3) launch the dashboard
streamlit run geomech_logml/app/streamlit_app.py
```

The app opens with a synthetic Agbada-like dataset pre-configured — press **🚂 Train & validate** and the full experiment (3 models × well-wise CV × interval validation) completes in ~1 minute on a laptop.

**What you can do in the dashboard (v0.3)**

- Upload your own **LAS/CSV** wells — with core targets (`E_STAT/NU_STAT/UCS/IS_CORE`) you train on your data; without them the app runs **transfer mode** (trains on synthetic Agbada data, predicts your wells, clearly flagged as indicative).
- **⚙️ Advanced hyperparameters** (sidebar): trees/depth/leaf for RF, rounds/depth/η for XGBoost, architecture/L2/iterations for the MLP.
- **Depth-window selector** on the prediction-curve tab for zooming into a reservoir zone.
- **Export**: curves + metrics (CSV), trained **model bundle (.joblib)**, and a **multi-page PDF report** (choose which wells appear as full-page curve tracks) + Markdown report.
- **Load trained model** (sidebar §1b): reuse an exported bundle on new wells instantly — prediction-only mode, no retraining.

**Docker alternative**

```bash
docker build -t geomech-logml .
docker run -p 8501:8501 geomech-logml
# → http://localhost:8501
```

**Command-line pipeline** (writes metrics, fold tables, interval coverage, ablation & report):

```bash
PYTHONPATH=. python scripts/train_eval.py --wells 8 --seed 42 --out-dir outputs/run
```

**Tests**

```bash
PYTHONPATH=. python -m pytest          # 39 tests, ~1 min
```

**Notebooks** (in `notebooks/`)

1. `01_synthetic_data_exploration.ipynb` — facies, overpressure, compaction, static–dynamic mismatch, core plugs.
2. `02_training_uncertainty_shap.ipynb` — training, well-wise CV, ablation, intervals, SHAP, report export.

---

## 2. Using your own data

Upload LAS or CSV files in the app (sidebar → *Upload LAS / CSV*). The loader normalises
mnemonics (`ILD/LLD/AT90→RT`, `DEN/RHOZ→RHOB`, `CNC/NPOR→NPHI`, `SGR/CGR→GR`, and
`DTC/DT→VP` via `VP = 10⁶/DTC`).

Expected curves:

| Curve | Required | Units | Notes |
|---|---|---|---|
| `DEPT` | yes | m | any depth mnemonic maps (`DEPTH`, `MD`, `TVD`…) |
| `GR` | yes | API | |
| `RHOB` | yes | g/cc | |
| `NPHI` | yes | v/v | |
| `RT` | yes | ohm·m | deep resistivity |
| `VP` | no | m/s | or `DTC` in µs/ft — auto-converted |
| `E_STAT, NU_STAT, UCS, IS_CORE` | no | GPa, –, MPa, 0/1 | supply **paired core measurements** to train on your own wells (see format in `examples/agbada_synthetic/*.csv`) |

If your file has no core targets, the app trains on the synthetic dataset and predicts your
uploaded curves (transfer mode — treat results as indicative until core-calibrated).

Example files live in `examples/agbada_synthetic/` (3 wells, LAS + CSV, known ground truth).

---

## 3. How the methodological gaps from the literature review are closed

| # | Gap identified in the literature | GeoMech-LogML implementation |
|---|---|---|
| 1 | Studies require shear-sonic/dipole data that legacy wells don't have | Input whitelist is exactly **GR, RHOB, NPHI, Rt (+ optional Vp)**; Vp-dependent features auto-drop; a dedicated **no-Vp feature set** is a first-class ablation arm |
| 2 | Random sample splits inflate accuracy on autocorrelated log data | **WellKFold / LeaveOneWellOut** — every metric is pooled out-of-fold over whole wells; even the MLP's internal early stopping (a random split) is disabled |
| 3 | Static↔dynamic conversion done as a fixed post-hoc factor | The conversion is **learned inside the model** from paired core rows; no factor exists anywhere in the codebase |
| 4 | Single-model studies, no family comparison | **RF vs XGBoost vs MLP** trained/evaluated identically and reported side-by-side per target |
| 5 | Point predictions with no uncertainty | **Split-conformal** intervals calibrated *well-wise* (calibration wells held out inside each fold) + **QRF** leaf-quantile intervals; empirical coverage on blind wells reported in-app |
| 6 | Black-box models, no interpretability | **SHAP** for every model family: beeswarm summary, dependence plots, per-depth waterfall explanations |
| 7 | No public paired datasets for weak siliciclastics | **Physics-based Agbada generator** with published-range constraints (Athy compaction w.r.t. effective stress, Hottman–Johnson-style overpressure ramp, Archie resistivity, Han velocities, porosity/clay-controlled UCS) — with ground-truth statics for honest end-to-end testing |
| 8 | Irreproducible experiments | Global seed threading, deterministic generator (test-enforced), pinned `requirements.txt`, Dockerfile with smoke-test + healthcheck, pytest suite, CLI, notebooks |

---

## 4. Synthetic data generator (what it models)

`geomech_logml/data/synthetic.py` — every step is vectorised, deterministic, and documented
with parameter values in the `SyntheticConfig` dataclass:

```
depth grid ─► Markov facies (sand/shaly sand/sandy shale/shale)
          ─► clay volume ─► pore pressure (hydrostatic → undercompaction ramp,
                             λ → 0.72–0.82·Sv below a 2.4–3.1 km top)
          ─► effective stress ─► Athy porosity (φ₀·e^(−kσ'))  [+ deep-sand cementation]
          ─► logs: GR (endmember mixing) · RHOB (volume mixing, HC dilution)
                   NPHI (HI mixing + shale bound water) · RT (parallel Archie/shale,
                   n=2, pay Sw 0.12–0.55) · VP (Han et al. 1986 + (σ'/40 MPa)^β pressure law)
          ─► geomechanics: Vs (hidden!) → Edyn, νdyn →
             E_STAT = Edyn·f(rock quality)  [f varies 0.25–0.85, depth-correlated noise]
             NU_STAT ≈ 0.96·νdyn            UCS = φ/clay-controlled + stress factor
          ─► IS_CORE flags = paired lab plugs (training data)
```

Physics checks are enforced in `tests/test_synthetic.py` (compaction vs σ′, overpressure →
porosity preservation, clay weakening at equal φ, static/dynamic ratio variability,
Agbada-range compliance, determinism).

**Reference physical models:** Athy (1930) compaction; Archie (1942); Hottman & Johnson
(1965) overpressure; Han, Nur & Morgan (1986) velocities; Larionov (1969) Vshale;
Meinshausen (2006) QRF; split-conformal after Papadopoulos et al. (2002)/Vovk et al.
(2005); property ranges cross-checked against Zoback (2007), Fjaer et al. (2008) and
published Niger Delta geomechanics studies. *The synthetic dataset is a research
construction, not a substitute for field calibration.*

---

## 5. Package structure

```
GeoMech-LogML/
├── geomech_logml/
│   ├── config.py               # single source of truth (columns, targets, bounds, seeds)
│   ├── data/                   # synthetic generator + LAS/CSV I/O (lasio + fallback)
│   ├── preprocessing/          # cleaning, petrophysical features, well-wise CV splitters
│   ├── models/                 # registry (RF/XGB/MLP) + OOF training/evaluation
│   ├── uncertainty/            # QRF + well-wise split conformal
│   ├── interpretability/       # SHAP wrappers (Tree/Kernel)
│   ├── pipeline.py             # run_experiment / run_ablation / interval validation
│   └── app/                    # Streamlit app + plots + report builder
├── notebooks/                  # 01 data exploration · 02 training/uncertainty/SHAP
├── scripts/train_eval.py       # CLI end-to-end pipeline
├── tests/                      # 39 pytest tests (physics, CV integrity, coverage, app)
├── examples/agbada_synthetic/  # 3 committed example wells (LAS + CSV, known truth)
├── requirements.txt · pyproject.toml · Dockerfile · .streamlit/config.toml
├── README.md · ADR.md          # this file · one-page architecture decisions
```

---

## 6. Method summary (for reviewers)

- **Task:** multi-target regression (E_STAT, NU_STAT, UCS) from ≤12 engineered features.
- **Training data:** core-plug rows (`IS_CORE=1`) — the only rows where static lab values exist.
- **Validation:** `WellKFold` (k = min(5, #wells)) or `LeaveOneWellOut`; pooled out-of-fold
  R²/RMSE/MAE per target; fold tables exported.
- **Intervals:** (a) split-conformal — nested *calibration wells* inside each fold,
  pooled-OOF residual quantile with finite-sample correction; (b) QRF — RF leaf-sample
  quantiles (interval forests use `min_samples_leaf=10`). Both validated on blind wells.
- **Explainability:** TreeSHAP (RF/XGB) & KernelSHAP (MLP); mean-|SHAP| feature ranking;
  beeswarm/dependence/waterfall exported.
- **Ablation:** `eng_with_vp` vs `eng_no_vp` (and raw variants) — ΔR²/ΔRMSE per model/target.

Typical blind-well results on 8 synthetic wells (seed 42): E R² ≈ 0.96–0.97,
ν R² ≈ 0.93, UCS R² ≈ 0.83–0.86; 90 % conformal coverage ≈ 0.83–0.91.

---

## 7. Limitations & intended use

- Research prototype: **not** a substitute for a core-calibrated geomechanics program.
- Synthetic training data encodes the generator's physics; real deployments should supply
  field core data (the app trains on uploaded paired data when provided).
- Inter-well geological shifts limit transfer — visible honestly as slight interval
  under-coverage (see Uncertainty tab); QRF does not absorb inter-well bias by design.
- Property predictions are clipped to physical bounds; intervals, not points, should drive
  decisions (e.g., sanding-risk screening, mud-weight windows).

## 8. License & citation

MIT (c) 2026 GeoMech-LogML contributors. If you use this work, cite the repository and the
upstream methods listed in §4.
