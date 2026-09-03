"""GeoMech-LogML — Streamlit application entry point.

Run with:
    streamlit run geomech_logml/app/streamlit_app.py

A non-programmer interface for petroleum engineers:
upload LAS/CSV (or generate Agbada-like synthetic data), train RF / XGBoost / MLP
under **well-wise** cross-validation, view continuous property curves with
uncertainty bands, inspect SHAP explanations, compare models side-by-side and
export CSV + a markdown report.
"""

from __future__ import annotations

import time

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

from geomech_logml.config import (
    CORE_FLAG,
    DEPTH_COL,
    GLOBAL_SEED,
    TARGETS,
    TARGET_LABELS,
    WELL_COL,
)
from geomech_logml.data.las_io import load_any
from geomech_logml.data.synthetic import SyntheticConfig, generate_dataset
from geomech_logml.models.registry import DEFAULT_MODELS, MODEL_SPECS
from geomech_logml.pipeline import (
    ExperimentConfig,
    ExperimentResult,
    interval_coverage_summary,
    qrf_curve_intervals,
    run_ablation,
    run_experiment,
)
from geomech_logml.preprocessing.features import (
    FEATURE_SETS,
    clean_logs,
    engineer_features,
)
from geomech_logml.interpretability.shap_explainer import ShapExplainer
from geomech_logml.app.plots import (
    ablation_delta_figure,
    coverage_scatter_figure,
    crossplot_figure,
    log_track_figure,
    oof_crossplot_grid,
    per_well_coverage_figure,
    per_well_metric_heatmap,
    prediction_track_figure,
    static_dynamic_figure,
)
from geomech_logml.app.pdf_report import build_pdf_bytes, pdf_available
from geomech_logml.app.report import build_report

matplotlib.use("Agg")  # headless-safe rendering for SHAP figures

# ---------------------------------------------------------------------------
# Page & branding
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoMech-LogML",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 0rem;}
      div[data-testid="stMetric"] {background:#f4f7fa; border:1px solid #dfe6ee;
        border-radius:8px; padding:10px 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached heavy operations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating Agbada-like synthetic dataset ...")
def _generate_cached(n_wells: int, seed: int, step: float, overpressure: bool) -> pd.DataFrame:
    cfg = SyntheticConfig(n_wells=int(n_wells), step_m=float(step),
                          overpressure=bool(overpressure), seed=int(seed))
    return generate_dataset(cfg)


@st.cache_resource(show_spinner=False)
def _train_experiment(
    data: pd.DataFrame, feature_set: str, models: tuple[str, ...],
    cv_strategy: str, n_splits: int, alpha: float, seed: int,
    fast_mlp: bool, validate_uncertainty: bool,
) -> ExperimentResult:
    overrides = {"mlp": {"max_iter": 1200, "hidden_layer_sizes": (48, 24)}} if fast_mlp else None
    cfg = ExperimentConfig(
        feature_set=feature_set, model_keys=list(models), cv_strategy=cv_strategy,
        n_splits=int(n_splits), alpha=float(alpha), seed=int(seed),
        hyper_overrides=overrides, validate_uncertainty=validate_uncertainty)
    return run_experiment(data, cfg)


# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------
st.sidebar.title("⛰️ GeoMech-LogML")
st.sidebar.caption("Rock strength & elastic properties from standard wireline logs")
st.sidebar.header("1 · Data source")

data_mode = st.sidebar.radio("Choose data", ["Synthetic Agbada-like", "Upload LAS / CSV"],
                             label_visibility="collapsed")

data: pd.DataFrame | None = None
data_source_label = ""

if data_mode == "Synthetic Agbada-like":
    c1, c2 = st.sidebar.columns(2)
    n_wells = c1.number_input("Wells", 4, 15, 6, help="More wells → better blind-well accuracy")
    seed = c2.number_input("Seed", 0, 9999, GLOBAL_SEED, help="Reproducibility seed")
    c3, c4 = st.sidebar.columns(2)
    step = c3.select_slider("Depth step (m)", options=[0.25, 0.5, 1.0, 2.0], value=1.0)
    overpressure = c4.checkbox("Overpressure", value=True,
                               help="Agbada-style pore-pressure ramp below ~2.4–3.1 km")
    data = _generate_cached(int(n_wells), int(seed), float(step), bool(overpressure))
    data_source_label = f"synthetic Agbada-like ({n_wells} wells, seed={seed})"
else:
    up = st.sidebar.file_uploader(
        "Upload well files", type=["las", "csv"], accept_multiple_files=True,
        help="LAS or CSV with columns among: DEPT, GR, RHOB, NPHI, RT, (optional) VP or DTC. "
             "Optionally E_STAT / NU_STAT / UCS + IS_CORE to train on your own paired data.")
    if up:
        frames = []
        for f in up:
            try:
                frames.append(load_any(f))
            except Exception as e:  # noqa: BLE001 — surface a readable error to the user
                st.sidebar.error(f"Could not read {f.name}: {e}")
        if frames:
            data = pd.concat(frames, ignore_index=True)
            data_source_label = f"uploaded ({len(up)} file(s))"
    else:
        st.info("Upload one or more LAS/CSV files, or switch to the synthetic dataset to "
                "explore the workflow immediately.")
        st.stop()

# QC the loaded data
missing = [c for c in ["GR", "RHOB", "NPHI", "RT"] if c not in data.columns]
if missing:
    st.error(f"Uploaded data is missing required logs: {missing}. "
             f"Found: {list(data.columns)}. See the README for the expected format.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — training controls
# ---------------------------------------------------------------------------
st.sidebar.header("2 · Training setup")

fs_labels = {k: v["label"] for k, v in FEATURE_SETS.items()}
default_fs = "eng_with_vp" if "VP" in data.columns else "eng_no_vp"
feature_set = st.sidebar.selectbox(
    "Feature set", list(FEATURE_SETS.keys()), index=list(FEATURE_SETS).index(default_fs),
    format_func=lambda k: fs_labels[k],
    help="Ablation sets — raw vs engineered petrophysical features, with/without Vp.")
if FEATURE_SETS[feature_set]["requires_vp"] and "VP" not in data.columns:
    st.sidebar.warning("No Vp in this dataset → Vp features are dropped automatically.")

model_keys = st.sidebar.multiselect(
    "Models to train", list(MODEL_SPECS.keys()), default=DEFAULT_MODELS,
    format_func=lambda k: MODEL_SPECS[k].label)

_CV_LABELS = {"well_kfold": "K-fold on wells (recommended)",
              "leave_one_well_out": "Leave-one-well-out"}
cv_strategy = st.sidebar.selectbox("Cross-validation", ["well_kfold", "leave_one_well_out"],
                                   format_func=lambda s: _CV_LABELS[s])
n_splits = st.sidebar.slider("Folds (wells per fold group)", 2, 10, 5,
                             disabled=(cv_strategy == "leave_one_well_out"))
alpha = st.sidebar.slider("Interval miscoverage α (0.10 = 90% PIs)",
                          0.05, 0.30, 0.10, 0.05, format="%.2f",
                          help="Prediction intervals target 1−α coverage "
                               "(α = 0.10 → 90% intervals).")
fast_mlp = st.sidebar.checkbox("Fast MLP mode (fewer iterations)", value=True,
                               help="Quicker training; uncheck for final runs.")
train_btn = st.sidebar.button("🚂 Train & validate", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# Session state & experiment execution
# ---------------------------------------------------------------------------
if train_btn and model_keys:
    t0 = time.perf_counter()
    with st.spinner(f"Training {len(model_keys)} model(s) under well-wise CV …"):
        result = _train_experiment(
            data, feature_set, tuple(model_keys), cv_strategy, n_splits,
            float(alpha), GLOBAL_SEED, fast_mlp, True)
    st.session_state["result"] = result
    st.session_state["result_sig"] = (feature_set, tuple(model_keys), cv_strategy,
                                      n_splits, float(alpha), len(data))
    st.toast(f"Training complete in {time.perf_counter() - t0:.0f}s", icon="✅")

result: ExperimentResult | None = st.session_state.get("result")
if result is not None:
    st.session_state["result_sig"] = st.session_state.get("result_sig")

# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------
st.title("⛰️ GeoMech-LogML")
st.caption("ML prediction of static Young's modulus, Poisson's ratio and UCS from "
           "standard wireline logs — built for unconsolidated siliciclastics "
           "(Agbada Formation, Niger Delta)")

qc = clean_logs(data)
qc_eng = engineer_features(qc)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wells loaded", qc[WELL_COL].nunique())
m2.metric("Log rows (cleaned)", f"{len(qc):,}")
m3.metric("Core-plug rows", f"{int(qc.get(CORE_FLAG, pd.Series(dtype=int)).sum()):,}")
m4.metric("Vp available", "yes" if "VP" in qc.columns else "no — legacy mode")

if result is None:
    st.info("👈 Configure the experiment in the sidebar and press **Train & validate** "
            "to see results. Everything is computed with **well-wise** cross-validation — "
            "models are always tested on wells they have never seen.")

tab_labels = ["📊 Data & QC", "🎯 Training & CV", "📈 Prediction curves",
              "🌀 Uncertainty", "🔍 SHAP", "🏆 Compare & Export"]
tabs = st.tabs(tab_labels)
_NOT_TRAINED = "👈 Train & validate in the sidebar to enable this tab."

# ===========================================================================
# TAB 1 — Data & QC
# ===========================================================================
with tabs[0]:
    st.subheader("Data overview")
    st.markdown(
        "Input logs and ground-truth geomechanics per well. If you are using the "
        "synthetic dataset, `E_STAT/NU_STAT/UCS` are the **known truth** (from the "
        "physics generator) and `IS_CORE` marks the subsample used for training — "
        "mimicking a real core-calibration dataset.")
    wells = sorted(qc[WELL_COL].unique())
    sel_well = st.selectbox("Well", wells, key="qc_well")
    st.plotly_chart(log_track_figure(qc_eng, sel_well), width="stretch")

    st.markdown("#### Petrophysical crossplots")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(crossplot_figure(qc_eng, "RHOB", "NPHI", color="VSH_TRUE"),
                        width="stretch")
    with cc2:
        st.plotly_chart(crossplot_figure(qc_eng, "VP", "VSH_TRUE", color="PHI_TRUE"),
                        width="stretch")

    st.markdown("#### Log statistics (per well)")
    stat_cols = [c for c in ["GR", "RHOB", "NPHI", "RT", "VP"] if c in qc.columns]
    st.dataframe(qc.groupby(WELL_COL)[stat_cols].agg(["mean", "std"]).round(3),
                 width="stretch")

    with st.expander("Overpressure QC (synthetic data only)"):
        if "PP_MPA" in qc.columns:
            import plotly.graph_objects as go
            w = qc_eng[qc_eng[WELL_COL] == sel_well]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w["PP_MPA"], y=w[DEPTH_COL], name="Pore pressure",
                                     line=dict(color="#B4491F")))
            fig.add_trace(go.Scatter(x=w["SV_MPA"], y=w[DEPTH_COL], name="Overburden (Sv)",
                                     line=dict(color="#333")))
            fig.add_trace(go.Scatter(x=0.0101 * w[DEPTH_COL], y=w[DEPTH_COL],
                                     name="Hydrostatic", line=dict(dash="dash", color="#7FA6C9")))
            fig.update_yaxes(autorange="reversed", title="Depth (m)")
            fig.update_xaxes(title="Pressure (MPa)")
            fig.update_layout(height=560, title=f"Pressure system — {sel_well}")
            st.plotly_chart(fig, width="stretch")

# ===========================================================================
# TAB 2 — Training & CV
# ===========================================================================
with tabs[1]:
    if result is None:
        st.info(_NOT_TRAINED)
        st.stop()
    st.subheader("Blind-well performance (pooled out-of-fold predictions)")
    st.markdown("Every metric below is computed on **wells entirely held out of training** "
                "(spatially independent validation — never random row splits).")
    st.dataframe(result.metrics, width="stretch", hide_index=True)

    import plotly.graph_objects as go
    fig = go.Figure()
    for key in result.config.model_keys:
        sub = result.metrics[result.metrics["ModelKey"] == key]
        fig.add_trace(go.Bar(x=sub["Target"], y=sub["R2"], name=MODEL_SPECS[key].label))
    fig.update_layout(barmode="group", yaxis_title="Blind-well R²",
                      yaxis_range=[min(0.0, result.metrics["R2"].min() - 0.05), 1.0],
                      height=380, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Blind-well crossplots — truth vs prediction (every model × target)")
    st.plotly_chart(oof_crossplot_grid(result), width="stretch")

    st.markdown("#### Per-well blind R²")
    st.plotly_chart(per_well_metric_heatmap(result), width="stretch")
    st.caption("Each cell is an independent blind test: that well was fully held out "
               "of training. Cool cells reveal geology the model has not seen — "
               "candidates for more calibration core.")

    with st.expander("Evidence: the static–dynamic conversion is learned, not fixed"):
        if "E_DYN" in result.data.columns:
            st.plotly_chart(static_dynamic_figure(result.data), width="stretch")
            st.caption("The paired core data define a *curve* (binned mean, orange), "
                       "not a constant ratio — exactly why GeoMech-LogML regresses "
                       "static properties directly from logs instead of applying a "
                       "hard-coded factor (grey dotted line shows why that fails).")
        else:
            st.info("E_DYN requires density + sonic data; not present in this dataset.")

    sel_model_fold = st.selectbox("Model (fold detail)",
                                  result.config.model_keys,
                                  format_func=lambda k: MODEL_SPECS[k].label,
                                  key="fold_model")
    st.markdown("#### Per-fold metrics")
    st.dataframe(result.cv[sel_model_fold].fold_metrics, width="stretch", hide_index=True)

    st.markdown(f"*Total runtime: {result.runtime_seconds:.0f}s · "
                f"training rows: {len(result.X_core)} core plugs from "
                f"{result.groups.nunique()} wells · feature set: "
                f"`{result.config.feature_set}` ({', '.join(result.feature_names)})*")

# ===========================================================================
# TAB 3 — Prediction curves
# ===========================================================================
with tabs[2]:
    if result is None:
        st.info(_NOT_TRAINED)
        st.stop()
    st.subheader("Continuous property curves with uncertainty bands")
    pcols = st.columns(4)
    well_opts = sorted(result.curves[WELL_COL].unique())
    curve_well = pcols[0].selectbox("Well", well_opts, key="curve_well")
    curve_model = pcols[1].selectbox("Model", result.config.model_keys,
                                     format_func=lambda k: MODEL_SPECS[k].label,
                                     key="curve_model")
    show_qrf = pcols[2].checkbox("Add QRF band (Random Forest)",
                                 disabled=("random_forest" not in result.config.model_keys),
                                 help="Quantile Regression Forest intervals — per-row, "
                                      "wider where the training data disagree.")
    depth_only = pcols[3].empty()

    curves = result.curves
    if show_qrf:
        with st.spinner("Computing QRF bands for this well …"):
            w_mask = (curves[WELL_COL] == curve_well)
            feat_cols = result.feature_names
            X_curve = result.data.loc[result.data[WELL_COL] == curve_well, feat_cols]
            X_curve = X_curve.astype(float)
            qrf = qrf_curve_intervals(result.X_core, result.Y_core, X_curve,
                                      alpha=result.config.alpha, seed=result.config.seed)
            for col in qrf.columns:
                curves.loc[w_mask, col] = qrf[col].to_numpy()
    st.plotly_chart(
        prediction_track_figure(curves, curve_well, curve_model,
                                include_qrf=show_qrf, alpha=result.config.alpha),
        width="stretch")

    st.caption("Grey = ground truth (synthetic) · blue = prediction · shaded = conformal "
               "90% prediction interval · dashed orange = QRF interval · red diamonds = "
               "core plugs used in training.")

# ===========================================================================
# TAB 4 — Uncertainty
# ===========================================================================
with tabs[3]:
    if result is None:
        st.info(_NOT_TRAINED)
        st.stop()
    st.subheader("Prediction-interval validation (honest, out-of-well)")
    st.markdown("Both interval methods are validated on wells that were **never seen** "
                "during training or calibration. Coverage should match the nominal level; "
                "widths communicate where the model is uncertain.")
    cov = interval_coverage_summary(result)
    st.dataframe(cov.style.format({"Nominal": "{:.0%}", "Coverage": "{:.1%}",
                                   "MeanWidth": "{:.3f}"}),
                 width="stretch", hide_index=True)

    method_map = {"QRF (out-of-well)": result.qrf_oof,
                  "Conformal (nested well-wise)": result.honest_conformal}
    u1, u2 = st.columns(2)
    with u1:
        t_sel = st.selectbox("Target", TARGETS, format_func=lambda t: TARGET_LABELS[t],
                             key="unc_target")
    with u2:
        src = st.selectbox("Interval source", list(method_map.keys()), key="unc_method")
    if method_map.get(src) is not None:
        st.plotly_chart(coverage_scatter_figure(method_map[src], t_sel,
                                                1 - result.config.alpha),
                        width="stretch")
        st.markdown("#### Per-well coverage")
        st.plotly_chart(per_well_coverage_figure(method_map[src], t_sel,
                                                 1 - result.config.alpha),
                        width="stretch")
        st.caption("Wells below the dashed nominal line have systematically larger "
                   "errors — real geological heterogeneity, quantified honestly.")
    st.markdown("#### OOF residual distribution")
    res_df = result.cv[result.config.model_keys[0]].oof
    res_df = res_df[res_df["TARGET"] == t_sel]
    import plotly.graph_objects as go
    fig = go.Figure(go.Histogram(x=(res_df["PRED"] - res_df["TRUE"]), nbinsx=40,
                                 marker_color="#1A5FB4"))
    fig.add_vline(x=0, line_dash="dash", line_color="#B4491F")
    fig.update_layout(height=340, xaxis_title="Prediction − truth",
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

# ===========================================================================
# TAB 5 — SHAP
# ===========================================================================
with tabs[4]:
    if result is None:
        st.info(_NOT_TRAINED)
        st.stop()
    st.subheader("SHAP explanations")
    st.markdown("Global beeswarm summary, dependence plots for the top features, and "
                "per-depth waterfall explanations. Tree models use exact TreeSHAP; the "
                "MLP uses a KernelSHAP approximation over a k-means background.")

    s1, s2, s3 = st.columns(3)
    shap_model = s1.selectbox("Model", result.config.model_keys,
                              format_func=lambda k: MODEL_SPECS[k].label, key="shap_model")
    shap_target = s2.selectbox("Target", TARGETS, format_func=lambda t: f"{TARGET_LABELS[t]} ({t})",
                               key="shap_target")
    n_explain = s3.slider("Rows to explain", 50, 500, 200, 50,
                          help="More rows = smoother plots, slower for the MLP.")

    cache_key = ("shap", shap_model, shap_target, n_explain)
    if st.button("🔍 Compute SHAP", key="shap_btn", type="primary"):
        with st.spinner("Explaining model … (KernelSHAP for the MLP may take a minute)"):
            rng = np.random.default_rng(0)
            idx = rng.choice(len(result.X_core), size=min(n_explain, len(result.X_core)),
                             replace=False)
            X_explain = result.X_core.iloc[idx]
            explainer = ShapExplainer(
                shap_model, shap_target,
                result.final_models[shap_model][shap_target],
                background=result.X_core.iloc[rng.choice(len(result.X_core),
                                                        size=min(300, len(result.X_core)),
                                                        replace=False)],
                seed=result.config.seed)
            vals = explainer.shap_values(X_explain)
            st.session_state[cache_key] = (explainer, X_explain, vals)
            st.rerun()

    if cache_key in st.session_state:
        explainer, X_explain, vals = st.session_state[cache_key]
        cA, cB = st.columns([3, 2])
        with cA:
            fig_sum = explainer.summary_figure(X_explain)
            st.pyplot(fig_sum, clear_figure=True)
        with cB:
            top = explainer.top_features(X_explain, k=4)
            st.markdown("**Top features by mean |SHAP|**")
            st.dataframe(pd.DataFrame({"rank": range(1, len(top) + 1), "feature": top}),
                         hide_index=True, width="stretch")
            st.markdown("Feature conventions: `VSH_GR` shale volume (Larionov), "
                        "`PHID/PHIND/PHIS` density/neutron/sonic porosity, "
                        "`AI` impedance, `RT_LOG10` log-resistivity, `DEPT` depth.")

        st.markdown("#### Dependence plots")
        dcols = st.columns(3)
        for i, feat in enumerate(top[:3]):
            with dcols[i]:
                fig_dep = explainer.dependence_figure(X_explain, feat)
                st.pyplot(fig_dep, clear_figure=True)

        st.markdown("#### Global feature importance (mean |SHAP|)")
        fig_bar = explainer.mean_abs_shap_figure(X_explain)
        st.pyplot(fig_bar, clear_figure=True)

        st.markdown("#### Per-depth explanation (waterfall)")
        w_sel, d_sel = st.columns(2)
        well_pick = w_sel.selectbox("Well", sorted(result.data[WELL_COL].unique()),
                                    key="wf_well")
        rows_wf = result.data[result.data[WELL_COL] == well_pick]
        d_idx = d_sel.selectbox("Row (≈depth)",
                                range(0, len(rows_wf), 25),
                                index=min(10, max(0, len(rows_wf) // 25 - 1)),
                                format_func=lambda i: f"{rows_wf.iloc[i][DEPTH_COL]:.0f} m",
                                key="wf_row")
        row_series = rows_wf.iloc[d_idx][result.feature_names].astype(float)
        fig_wf = explainer.waterfall_figure(row_series)
        st.pyplot(fig_wf, clear_figure=True)
    else:
        st.info("Press **Compute SHAP** to generate explanations for the selected "
                "model/target pair.")

# ===========================================================================
# TAB 6 — Compare & export
# ===========================================================================
with tabs[5]:
    if result is None:
        st.info(_NOT_TRAINED)
        st.stop()
    st.subheader("Side-by-side model comparison")
    pivot = result.metrics.pivot(index="Model", columns="Target", values="R2")
    st.dataframe(pivot.style.highlight_max(axis=0, color="#c8e6c9").format("{:.3f}"),
                 width="stretch")
    st.caption("Cells show blind-well R². Green = best model per target.")

    st.markdown("#### Feature-set ablation — does Vp help?")
    abl_key = ("ablation", result.config.seed, result.config.cv_strategy)
    if st.button("⚗️ Run ablation (with vs without Vp)", key="abl_btn"):
        with st.spinner("Re-running both feature sets under well-wise CV …"):
            abl = run_ablation(result.data, result.config)
            st.session_state[abl_key] = abl
    if abl_key in st.session_state:
        abl = st.session_state[abl_key]
        st.dataframe(abl, width="stretch", hide_index=True)
        st.plotly_chart(ablation_delta_figure(abl), width="stretch")
        st.caption("`dR2_Vp` > 0 means the sonic log improved blind-well accuracy. "
                   "Even without Vp the models remain usable — that is the point of the "
                   "legacy-suite feature set.")

    st.markdown("---")
    st.subheader("Export")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button("⬇️ Curve predictions (CSV)",
                           result.curves.to_csv(index=False).encode(),
                           file_name="geomech_curve_predictions.csv", mime="text/csv")
    with e2:
        st.download_button("⬇️ Metrics (CSV)",
                           result.metrics.to_csv(index=False).encode(),
                           file_name="geomech_metrics.csv", mime="text/csv")

    st.markdown("#### Full report")
    if pdf_available():
        if st.button("🧾 Generate PDF report (charts + tables)", key="pdf_btn",
                     help="Multi-page PDF: summary, methodology, crossplots, "
                          "interval validation, example curves, SHAP."):
            with st.spinner("Rendering PDF report …"):
                try:
                    st.session_state["pdf_bytes"] = build_pdf_bytes(result, data_source_label)
                except Exception as e:  # noqa: BLE001
                    st.error(f"PDF generation failed: {e}")
        if "pdf_bytes" in st.session_state:
            st.download_button(
                "⬇️ Download report (PDF)",
                st.session_state["pdf_bytes"],
                file_name="geomech_report.pdf", mime="application/pdf")
            st.caption(f"Size: {len(st.session_state['pdf_bytes']) / 1024:.0f} KB")
    else:
        st.warning("Install `reportlab` to enable PDF export "
                   "(pip install reportlab). Markdown export is still available.")
    st.download_button("⬇️ Report (Markdown)",
                       build_report(result, data_source_label).encode(),
                       file_name="geomech_report.md", mime="text/markdown")

    st.markdown("#### Report preview")
    with st.expander("Show report"):
        st.markdown(build_report(result, data_source_label))

st.sidebar.divider()
st.sidebar.caption("GeoMech-LogML v0.2 · research prototype\n\n"
                   "Inputs: GR, RHOB, NPHI, RT (+ optional Vp) — no shear sonic "
                   "required anywhere in the workflow.")
