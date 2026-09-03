"""Markdown report generation for exports from the Streamlit app / CLI."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

from geomech_logml.config import TARGETS, TARGET_LABELS, TARGET_UNITS, WELL_COL
from geomech_logml.pipeline import ExperimentResult, interval_coverage_summary

__all__ = ["build_report", "save_report"]


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.to_string(index=False)


def build_report(res: ExperimentResult, data_source: str = "synthetic (Agbada-like)") -> str:
    """Compose a self-contained markdown report of one experiment run."""
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg = res.config
    lines: list[str] = []
    lines.append("# GeoMech-LogML — Geomechanical Prediction Report")
    lines.append("")
    lines.append(f"*Generated {now} · seed `{cfg.seed}` · data source: {data_source}*")
    lines.append("")

    # 1. Executive summary
    if res.metrics.empty:
        lines.append("> **PREDICTION-ONLY RUN** — predictions come from a loaded model "
                     "bundle. No cross-validation results are included; treat intervals "
                     "as indicative until core control is available.")
        lines.append("")
    elif res.curves_user is not None:
        lines.append("> **TRANSFER MODE** — the models were trained on the training "
                     "wells only and applied to uploaded wells without core control. "
                     "Metrics below refer to the training-domain (blind-well) "
                     "validation; the uploaded-well predictions carry the same "
                     "calibrated intervals but no ground-truth verification.")
        lines.append("")
    best = res.metrics.loc[res.metrics.groupby("Target")["R2"].idxmax()]
    lines.append("## 1. Executive summary")
    lines.append("")
    lines.append(f"- Feature set: **{cfg.feature_set}** ({len(res.feature_names)} features: "
                 f"{', '.join(res.feature_names)})")
    lines.append(f"- Validation: **{cfg.cv_strategy}** (whole wells held out — no random splits)")
    lines.append(f"- Wells in training data: {res.groups.nunique()} · core-plug rows: {len(res.X_core)}")
    for _, row in best.iterrows():
        lines.append(f"- Best model for **{row['Target']}**: {row['Model']} "
                     f"(blind-well R² = {row['R2']:+.3f}, RMSE = {row['RMSE']:.3f} "
                     f"{TARGET_UNITS[row['Target']]})")
    lines.append("")

    # 2. Methodology
    lines.append("## 2. Methodology checklist")
    lines.append("")
    lines.append("| Requirement | Implementation |")
    lines.append("|---|---|")
    lines.append("| Standard logs only (GR, RHOB, NPHI, RT, optional Vp) | "
                 "feature whitelist excludes any shear-derived quantity |")
    lines.append("| Targets: E_static, ν, UCS | per-target estimators, physical clipping |")
    lines.append("| Well-wise (spatially independent) CV | "
                 f"{cfg.cv_strategy}, whole wells per fold, MLP early-stopping disabled |")
    lines.append("| Static–dynamic conversion learned, not hard-coded | "
                 "models regress static properties directly from logs; no post-hoc factor |")
    lines.append(f"| Uncertainty: prediction intervals at {1-cfg.alpha:.0%} | "
                 "QRF + well-wise split conformal (calibration from held-out wells) |")
    lines.append("| Interpretability | SHAP summary + dependence + per-depth waterfall |")
    lines.append("")

    # 3. Metrics
    lines.append("## 3. Blind-well performance (pooled out-of-fold)")
    lines.append("")
    lines.append(_md_table(res.metrics))
    lines.append("")
    lines.append("### Per-fold detail")
    for key, cv in res.cv.items():
        lines.append("")
        lines.append(f"**{cv.model_key}** — fold metrics")
        lines.append("")
        lines.append(_md_table(cv.fold_metrics.drop(columns=["TEST_WELLS"])))
    lines.append("")

    # 4. Uncertainty validation
    cov = interval_coverage_summary(res)
    if not cov.empty:
        lines.append("## 4. Prediction-interval validation (honest, out-of-well)")
        lines.append("")
        lines.append(_md_table(cov))
        lines.append("")
        lines.append("*Coverage values near the nominal level indicate calibrated intervals; "
                     "deviations reflect genuine well-to-well heterogeneity (not row leakage).*")
        lines.append("")

    # 4b. Transfer predictions
    if res.curves_user is not None:
        lines.append("## 4b. Applied predictions (uploaded wells)")
        lines.append("")
        n_user = res.curves_user[WELL_COL].nunique()
        lines.append(f"- Uploaded wells predicted: **{n_user}** "
                     f"({', '.join(sorted(res.curves_user[WELL_COL].unique()))})")
        lines.append("- No ground truth available for these wells — intervals, not "
                     "points, should drive interpretation.")
        lines.append("- Predictions become decision-grade once core control "
                     "(even a few plugs) is supplied and the models retrained.")
        lines.append("")

    # 5. Usage guidance
    lines.append("## 5. Using the predictions")
    lines.append("")
    for t in TARGETS:
        lines.append(f"- **{t}** ({TARGET_LABELS[t]}, {TARGET_UNITS[t]}): predictions are "
                     f"clipped to physical bounds; use intervals to flag low-confidence "
                     f"intervals for coring/sanding-risk screening.")
    lines.append("")
    lines.append("---")
    lines.append("*GeoMech-LogML research prototype. Predictions are not a substitute for "
                 "core-calibrated geomechanical testing programs.*")
    return "\n".join(lines)


def save_report(res: ExperimentResult, path: str | Path,
                data_source: str = "synthetic (Agbada-like)") -> Path:
    p = Path(path)
    p.write_text(build_report(res, data_source), encoding="utf-8")
    return p
