"""Integration tests: SHAP, LAS I/O round-trip, pipeline experiment, Streamlit smoke."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pytest


# ---------------------------------------------------------------------------
# LAS I/O
# ---------------------------------------------------------------------------
def test_las_roundtrip(tmp_path, small_df):
    from geomech_logml.data.las_io import read_las, write_las

    well_df = small_df[small_df["WELL"] == small_df["WELL"].iloc[0]].head(500)
    path = tmp_path / "AGB-01.las"
    write_las(well_df, path)
    back = read_las(path)

    assert len(back) == len(well_df)
    for col in ("GR", "RHOB", "NPHI", "RT", "VP", "E_STAT", "UCS"):
        assert col in back.columns, f"LAS lost column {col}"
    np.testing.assert_allclose(
        back["GR"].to_numpy(), well_df["GR"].to_numpy(), atol=1e-3)


def test_csv_alias_mapping(tmp_path, small_df):
    """Legacy mnemonics (ILD, DEN, DT) must map to canonical columns."""
    from geomech_logml.data.las_io import load_any

    sub = small_df.head(50).copy()
    vp_true = sub["VP"].to_numpy().copy()
    sub = sub.rename(columns={"RT": "ILD", "RHOB": "DEN"})
    sub["DT"] = 1e6 / sub["VP"]
    sub = sub.drop(columns=["VP"])
    p = tmp_path / "legacy.csv"
    sub.to_csv(p, index=False)
    out = load_any(p)
    assert "RT" in out.columns and "RHOB" in out.columns
    np.testing.assert_allclose(out["VP"].to_numpy(), vp_true, rtol=1e-4)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def test_shap_tree_additivity(core_matrices):
    from geomech_logml.interpretability.shap_explainer import ShapExplainer
    from geomech_logml.models.registry import train_models

    X, Y, g = core_matrices
    bundle = train_models(X, Y, model_keys=["random_forest"], seed=0)
    ex = ShapExplainer("random_forest", "UCS", bundle["random_forest"]["UCS"],
                       background=X)
    row = X.iloc[[7]]
    vals = ex.shap_values(row)[0]
    pred = bundle["random_forest"]["UCS"].predict(row)[0]
    assert abs((ex.expected_value + vals.sum()) - pred) < 1e-4
    assert vals.shape[0] == X.shape[1]


def test_shap_figures_render(core_matrices):
    from geomech_logml.interpretability.shap_explainer import ShapExplainer
    from geomech_logml.models.registry import train_models

    X, Y, g = core_matrices
    bundle = train_models(X, Y, model_keys=["random_forest"], seed=0)
    ex = ShapExplainer("random_forest", "E_STAT", bundle["random_forest"]["E_STAT"],
                       background=X)
    Xs = X.iloc[:60]
    _ = ex.summary_figure(Xs); plt.close("all")
    _ = ex.dependence_figure(Xs, ex.top_features(Xs, k=1)[0]); plt.close("all")
    _ = ex.waterfall_figure(X.iloc[3]); plt.close("all")


# ---------------------------------------------------------------------------
# Pipeline (small end-to-end)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_result(engineered):
    from geomech_logml.pipeline import ExperimentConfig, run_experiment
    cfg = ExperimentConfig(
        feature_set="eng_no_vp", model_keys=["random_forest", "xgboost"],
        cv_strategy="well_kfold", n_splits=3, seed=0,
        hyper_overrides={}, validate_uncertainty=True)
    return run_experiment(engineered, cfg)


def test_experiment_outputs(tiny_result):
    assert set(tiny_result.cv) == {"random_forest", "xgboost"}
    assert len(tiny_result.metrics) == 6                      # 2 models × 3 targets
    assert tiny_result.honest_conformal is not None
    assert tiny_result.qrf_oof is not None
    for key in ("random_forest", "xgboost"):
        for t in ("E_STAT", "NU_STAT", "UCS"):
            assert f"{t}_{key}" in tiny_result.curves.columns
            assert f"{t}_{key}_LO" in tiny_result.curves.columns


def test_coverage_summary(tiny_result):
    from geomech_logml.pipeline import interval_coverage_summary
    cov = interval_coverage_summary(tiny_result)
    assert set(cov["Method"].unique()) == {"Conformal (nested well-wise)", "QRF (out-of-well)"}
    assert cov["Coverage"].between(0.5, 1.0).all()


def test_report_generation(tiny_result):
    from geomech_logml.app.report import build_report
    md = build_report(tiny_result)
    assert "GeoMech-LogML" in md and "blind-well" in md.lower()
    assert "R2" in md or "R²" in md


def test_pdf_report_generation(tiny_result):
    """PDF export builds a valid multi-page document with figures and tables."""
    pytest.importorskip("reportlab")
    from geomech_logml.app.pdf_report import build_pdf_bytes, pdf_available
    assert pdf_available()
    pdf = build_pdf_bytes(tiny_result, "synthetic test")
    assert pdf[:5] == b"%PDF-" and pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 50_000          # figures embedded
    assert pdf.count(b"/Type /Page") >= 4   # multi-page report


# ---------------------------------------------------------------------------
# Streamlit app smoke test (skipped if streamlit.testing unavailable)
# ---------------------------------------------------------------------------
def test_streamlit_app_smoke():
    stt = pytest.importorskip("streamlit.testing.v1")
    from pathlib import Path
    app_path = Path(__file__).resolve().parents[1] / "geomech_logml" / "app" / "streamlit_app.py"
    at = stt.AppTest.from_file(str(app_path), default_timeout=300)
    at.run()
    assert not at.exception, at.exception[0].stack_trace if at.exception else ""
    assert any("GeoMech-LogML" in t.value for t in at.title)
