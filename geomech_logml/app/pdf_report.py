"""Professional multi-page PDF report generation for GeoMech-LogML.

Built with reportlab (platypus) + matplotlib figure embeds — no system binaries
required (unlike wkhtmltopdf/WeasyPrint), so it works everywhere including Docker.

Sections:
  1. Cover / executive summary / configuration
  2. Methodology checklist (how each literature-review gap is closed)
  3. Blind-well performance: metrics table + crossplot grid figure
  4. Uncertainty: interval coverage table + per-target calibration figures
  5. Prediction curve example + SHAP global explanation
  6. Disclaimer / usage notes

Unicode note: reportlab's default Helvetica is Latin-1; Greek/arrows are
transliterated via :func:`_safe` to keep the PDF robust.
"""

from __future__ import annotations

import datetime as _dt
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from geomech_logml.config import TARGETS, TARGET_UNITS
from geomech_logml.models.registry import MODEL_SPECS

__all__ = ["build_pdf_bytes", "pdf_available"]

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _REPORTLAB = True
except ImportError:  # pragma: no cover - optional dependency
    _REPORTLAB = False


def pdf_available() -> bool:
    return _REPORTLAB


# ---------------------------------------------------------------------------
# Text safety (Helvetica = Latin-1 only)
# ---------------------------------------------------------------------------
_REPL = {"ν": "nu", "α": "alpha", "≈": "~", "→": "->", "−": "-", "·": "-",
         "×": "x", "≥": ">=", "≤": "<=", "τ": "tau", "σ": "sigma", "φ": "phi",
         "Δ": "Delta", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-"}


def _safe(s) -> str:
    return "".join(_REPL.get(ch, ch) for ch in str(s))


# ---------------------------------------------------------------------------
# Figure builders (matplotlib -> PNG bytes)
# ---------------------------------------------------------------------------
def _fig_to_bytes(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _crossplot_grid_png(result) -> io.BytesIO:
    model_keys = result.config.model_keys
    n_r, n_c = len(TARGETS), len(model_keys)
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.2 * n_c, 3.6 * n_r),
                             squeeze=False, sharex="row")
    for r, target in enumerate(TARGETS):
        for c, key in enumerate(model_keys):
            ax = axes[r][c]
            sub = result.cv[key].oof
            sub = sub[sub["TARGET"] == target]
            ax.scatter(sub["TRUE"], sub["PRED"], s=8, alpha=0.6, color="#1A5FB4")
            lo = float(min(sub["TRUE"].min(), sub["PRED"].min()))
            hi = float(max(sub["TRUE"].max(), sub["PRED"].max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
            r2 = r2_score(sub["TRUE"], sub["PRED"])
            rmse = root_mean_squared_error(sub["TRUE"], sub["PRED"])
            ax.set_title(f"{MODEL_SPECS[key].label} | {target}\n"
                         f"R2={r2:.3f}  RMSE={rmse:.2f} {TARGET_UNITS[target]}",
                         fontsize=8)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel("Prediction", fontsize=8)
            if r == n_r - 1:
                ax.set_xlabel("Lab truth", fontsize=8)
    fig.suptitle("Blind-well crossplots (pooled out-of-fold predictions)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _fig_to_bytes(fig)


def _coverage_png(result, alpha: float) -> io.BytesIO:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.6 * len(TARGETS), 4.0),
                             squeeze=False)
    for a, target in enumerate(TARGETS):
        ax = axes[0][a]
        if result.honest_conformal is not None:
            sub = result.honest_conformal[result.honest_conformal["TARGET"] == target]
            ax.errorbar(sub["PRED"], sub["TRUE"],
                        xerr=[sub["PRED"] - sub["LO"], sub["HI"] - sub["PRED"]],
                        fmt="o", ms=2.5, lw=0.7, elinewidth=0.5, alpha=0.7,
                        color="#1A5FB4", label="conformal PI")
        if result.qrf_oof is not None:
            sub2 = result.qrf_oof[result.qrf_oof["TARGET"] == target]
            ax.scatter(sub2["PRED"], sub2["TRUE"], s=4, alpha=0.35,
                       color="#B4491F", label="QRF (median)")
            lo_all = min(ax.get_xlim()[0], float(sub2["LO"].min()))
            hi_all = max(ax.get_xlim()[1], float(sub2["HI"].max()))
            ax.set_xlim(lo_all, hi_all)
        ax.set_title(target, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Prediction", fontsize=8)
        if a == 0:
            ax.set_ylabel("Lab truth", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(f"Interval calibration on blind wells (nominal {1 - alpha:.0%})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _fig_to_bytes(fig)


def _curve_track_png(result, well: str, model_key: str) -> io.BytesIO:
    curves = result.curves[result.curves["WELL"] == well].sort_values("DEPT")
    logs = result.data[result.data["WELL"] == well].sort_values("DEPT")
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 8.0), sharey=True)
    # input logs for context
    for ax, (col, ttl) in zip(axes[:2], [("GR", "GR (API)"), ("RHOB", "RHOB (g/cc)")]):
        ax.plot(logs[col], logs["DEPT"], lw=0.5, color="#20425A")
        ax.set_title(ttl, fontsize=9)
    for i, target in enumerate(TARGETS):
        ax = axes[2 + i]
        pred_col = f"{target}_{model_key}"
        ax.fill_betweenx(curves["DEPT"], curves[f"{pred_col}_LO"],
                         curves[f"{pred_col}_HI"], alpha=0.25, color="#1A5FB4",
                         label="90% PI")
        ax.plot(curves[pred_col], curves["DEPT"], lw=0.9, color="#1A5FB4",
                label="prediction")
        if f"{target}_TRUE" in curves.columns:
            ax.plot(curves[f"{target}_TRUE"], curves["DEPT"], lw=0.7,
                    color="#666666", label="truth")
            core = curves[curves.get("IS_CORE", 0) == 1]
            if len(core):
                ax.plot(core[f"{target}_TRUE"], core["DEPT"], "d", ms=4,
                        color="#E01B24", label="core")
        ax.set_title(f"{target} ({TARGET_UNITS[target]})", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="lower right")
    for ax in axes:
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Prediction curves with uncertainty — {well} "
                 f"({MODEL_SPECS[model_key].label})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _fig_to_bytes(fig)


def _shap_png(result) -> io.BytesIO | None:
    """Global SHAP beeswarm for the best tree model on UCS (fast TreeSHAP)."""
    try:
        import shap
        key = "random_forest" if "random_forest" in result.final_models else \
            next(k for k in result.final_models if
                 MODEL_SPECS[k].shap_kind == "tree")
        from geomech_logml.interpretability.shap_explainer import ShapExplainer
        rng = np.random.default_rng(0)
        idx = rng.choice(len(result.X_core), size=min(200, len(result.X_core)),
                         replace=False)
        X_explain = result.X_core.iloc[idx]
        ex = ShapExplainer(key, "UCS", result.final_models[key]["UCS"],
                           background=result.X_core)
        vals = ex.shap_values(X_explain)
        shap.summary_plot(vals, X_explain, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7.5, 4.2)
        fig.suptitle(f"SHAP summary (UCS) - {MODEL_SPECS[key].label}", y=1.02,
                     fontsize=10)
        return _fig_to_bytes(fig)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------
def _df_table(df: pd.DataFrame, styles, font_size: int = 7) -> Table:
    data = [[_safe(c) for c in df.columns]] + [
        [f"{v:.3f}" if isinstance(v, float) else _safe(v) for v in row]
        for _, row in df.iterrows()
    ]
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A5FB4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9C4CF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F7FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def build_pdf_bytes(result, data_source_label: str = "synthetic (Agbada-like)") -> bytes:
    """Render the full experiment report as a PDF and return its bytes."""
    if not _REPORTLAB:
        raise ImportError("reportlab is required for PDF export: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="GeoMech-LogML Geomechanical Prediction Report",
        author="GeoMech-LogML")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("H1x", parent=ss["Title"], fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("H2x", parent=ss["Heading2"], fontSize=13,
                        spaceBefore=12, spaceAfter=4,
                        textColor=colors.HexColor("#1A5FB4"))
    body = ParagraphStyle("Bodyx", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("Smallx", parent=body, fontSize=7.5, leading=9.5,
                           textColor=colors.HexColor("#4B5563"))
    cfg = result.config
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    story: list = []

    # ---- 1. Cover ----------------------------------------------------------
    story.append(Paragraph("GeoMech-LogML", h1))
    story.append(Paragraph("Geomechanical Prediction Report - rock strength &amp; "
                           "elastic properties from standard wireline logs", body))
    story.append(Spacer(1, 4))
    story.append(Paragraph(_safe(f"Generated {now} | seed {cfg.seed} | "
                                 f"data source: {data_source_label}"), small))
    story.append(Spacer(1, 10))

    best = result.metrics.loc[result.metrics.groupby("Target")["R2"].idxmax()]
    bullets = [
        f"Feature set: {cfg.feature_set} ({len(result.feature_names)} features: "
        + ", ".join(result.feature_names) + ")",
        f"Validation: {cfg.cv_strategy} - whole wells held out per fold "
        f"(no random splits; MLP early-stopping disabled).",
        f"Training data: {result.groups.nunique()} wells, {len(result.X_core)} "
        f"core-plug rows (paired static measurements).",
        f"Prediction intervals: QRF + well-wise split conformal at "
        f"{1 - cfg.alpha:.0%} nominal coverage.",
    ]
    for _, row in best.iterrows():
        bullets.append(f"Best model for {row['Target']}: {row['Model']} "
                       f"(blind-well R2 = {row['R2']:+.3f}, RMSE = "
                       f"{row['RMSE']:.3f} {TARGET_UNITS[row['Target']]})")
    for b in bullets:
        story.append(Paragraph("- " + _safe(b), body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Methodology checklist", h2))
    check = pd.DataFrame([
        ["Standard logs only (GR, RHOB, NPHI, RT, optional Vp)",
         "Feature whitelist excludes any shear-derived quantity"],
        ["Targets: E_static, Poisson ratio, UCS",
         "Per-target estimators; predictions clipped to physical bounds"],
        ["Well-wise (spatially independent) CV",
         f"{cfg.cv_strategy}: whole wells per fold; pooled out-of-fold metrics"],
        ["Static-dynamic conversion learned, not hard-coded",
         "Models regress statics directly from logs; no post-hoc factor"],
        [f"Uncertainty: {1 - cfg.alpha:.0%} prediction intervals",
         "QRF leaf quantiles + conformal bands calibrated on held-out wells"],
        ["Interpretability",
         "SHAP summary, dependence and per-depth waterfalls for every model"],
    ], columns=["Requirement", "Implementation"])
    story.append(_df_table(check, ss, font_size=7.5))
    story.append(PageBreak())

    # ---- 2. Performance ----------------------------------------------------
    story.append(Paragraph("1. Blind-well performance", h2))
    story.append(_df_table(result.metrics, ss))
    story.append(Spacer(1, 8))
    story.append(Image(_crossplot_grid_png(result), width=17.2 * cm,
                       height=17.2 * cm * 0.62))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Crossplots pool every out-of-fold prediction: each point "
                           "comes from a model that never saw that well during "
                           "training.", small))
    story.append(PageBreak())

    # ---- 3. Per-fold / per-well detail --------------------------------------
    story.append(Paragraph("2. Fold-level detail", h2))
    key0 = result.config.model_keys[0]
    story.append(Paragraph(_safe(f"Model: {MODEL_SPECS[key0].label}"), body))
    story.append(_df_table(result.cv[key0].fold_metrics.drop(columns=["TEST_WELLS"]), ss))
    story.append(Spacer(1, 10))

    from geomech_logml.pipeline import interval_coverage_summary
    cov = interval_coverage_summary(result)
    if not cov.empty:
        story.append(Paragraph("3. Prediction-interval validation (honest, "
                               "out-of-well)", h2))
        story.append(_df_table(cov.round(3), ss))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Coverage near the nominal level indicates calibrated "
                               "intervals; deviations reflect genuine well-to-well "
                               "heterogeneity (not row leakage). QRF does not absorb "
                               "inter-well bias by design - conformal is the primary "
                               "calibrated band.", small))
        story.append(Spacer(1, 8))
        story.append(Image(_coverage_png(result, cfg.alpha), width=17.2 * cm,
                           height=17.2 * cm * 0.42))
    story.append(PageBreak())

    # ---- 4. Curves + SHAP ----------------------------------------------------
    story.append(Paragraph("4. Example prediction curves", h2))
    well = sorted(result.curves["WELL"].unique())[0]
    story.append(Paragraph(_safe(
        f"Well {well}; blue = prediction, shaded = {1 - cfg.alpha:.0%} conformal band, "
        f"grey = ground truth (synthetic), red = core plugs."), body))
    story.append(Image(_curve_track_png(result, well, key0), width=17.2 * cm,
                       height=17.2 * cm * 0.78))
    shap_img = _shap_png(result)
    if shap_img is not None:
        story.append(PageBreak())
        story.append(Paragraph("5. Global SHAP explanation (UCS)", h2))
        story.append(Image(shap_img, width=16.5 * cm, height=16.5 * cm * 0.56))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Full dependence plots and per-depth waterfalls are "
                               "available in the dashboard (SHAP tab) and notebooks.",
                               small))

    # ---- footer ---------------------------------------------------------------
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "GeoMech-LogML research prototype. Predictions are not a substitute for "
        "core-calibrated geomechanical testing programs. Use prediction intervals, "
        "not point estimates, to support decisions (sanding risk, mud-weight "
        "windows, caprock integrity).", small))

    doc.build(story)
    return buf.getvalue()
