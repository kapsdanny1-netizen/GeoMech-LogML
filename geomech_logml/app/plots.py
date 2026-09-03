"""Plotting helpers for the Streamlit dashboard (plotly curves + matplotlib SHAP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, root_mean_squared_error

from geomech_logml.config import (
    DEPTH_COL,
    FACIES_NAMES,
    TARGETS,
    TARGET_LABELS,
    TARGET_UNITS,
    WELL_COL,
)

__all__ = [
    "log_track_figure",
    "prediction_track_figure",
    "crossplot_figure",
    "coverage_scatter_figure",
    "facies_colors",
]

#: Colorblind-friendly facies palette
FACIES_COLORS = {0: "#FDE725", 1: "#5EC962", 2: "#21918C", 3: "#3B528B"}


def facies_colors(series: pd.Series) -> list[str]:
    return [FACIES_COLORS.get(int(f), "#999999") for f in series]


def log_track_figure(df: pd.DataFrame, well: str) -> go.Figure:
    """Standard 5-track log display for one well (GR, RHOB, NPHI, RT, VP)."""
    w = df[df[WELL_COL] == well]
    tracks = [
        ("GR", "GR (API)", "y", False),
        ("RHOB", "RHOB (g/cc)", "y", False),
        ("NPHI", "NPHI (v/v)", "y", True),          # reversed porosity scale
        ("RT", "RT (ohm·m)", "log", False),
        ("VP", "VP (m/s)", "y", False),
    ]
    tracks = [t for t in tracks if t[0] in w.columns]
    fig = make_subplots(rows=1, cols=len(tracks), shared_yaxes=True,
                        subplot_titles=[t[1] for t in tracks])
    for i, (col, _label, _mode, reverse) in enumerate(tracks, start=1):
        fig.add_trace(
            go.Scattergl(x=w[col], y=w[DEPTH_COL], mode="lines",
                         line=dict(width=1.2, color="#20425A"), name=col,
                         showlegend=False),
            row=1, col=i)
        if reverse:
            fig.update_xaxes(autorange="reversed", row=1, col=i)
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
    fig.update_layout(height=700, title=f"Input logs — {well}",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def prediction_track_figure(
    curves: pd.DataFrame,
    well: str,
    model_key: str,
    include_qrf: bool = False,
    alpha: float = 0.10,
) -> go.Figure:
    """Depth tracks: truth + core plugs + model prediction with interval bands."""
    w = curves[curves[WELL_COL] == well].sort_values(DEPTH_COL)
    n = len(TARGETS) + (1 if "FACIES" in w.columns else 0)
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True,
                        subplot_titles=[f"{TARGET_LABELS[t]}" for t in TARGETS] +
                                       (["Facies"] if "FACIES" in w.columns else []))
    for i, target in enumerate(TARGETS, start=1):
        pred_col = f"{target}_{model_key}"
        lo_col, hi_col = f"{pred_col}_LO", f"{pred_col}_HI"
        if pred_col not in w.columns:
            continue
        if lo_col in w.columns:
            fig.add_trace(go.Scattergl(
                x=w[hi_col], y=w[DEPTH_COL], mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ), row=1, col=i)
            fig.add_trace(go.Scattergl(
                x=w[lo_col], y=w[DEPTH_COL], mode="lines", fill="tonexty",
                fillcolor="rgba(66,133,244,0.18)", line=dict(width=0),
                hoverinfo="skip", showlegend=False,
                name=f"{1-alpha:.0%} conformal band",
            ), row=1, col=i)
        # QRF band (dashed lines) if available
        if include_qrf and f"{target}_QLO" in w.columns:
            for c, dash in ((f"{target}_QHI", "dash"), (f"{target}_QLO", "dash")):
                fig.add_trace(go.Scattergl(
                    x=w[c], y=w[DEPTH_COL], mode="lines",
                    line=dict(width=1, color="#B4491F", dash=dash),
                    name=f"QRF {c.split('_')[0]} band", showlegend=False,
                ), row=1, col=i)
        truth_col = f"{target}_TRUE"
        if truth_col in w.columns:
            fig.add_trace(go.Scattergl(
                x=w[truth_col], y=w[DEPTH_COL], mode="lines",
                line=dict(width=1.0, color="#666666"), name="Truth",
                showlegend=(i == 1), legendgroup="truth",
            ), row=1, col=i)
        fig.add_trace(go.Scattergl(
            x=w[pred_col], y=w[DEPTH_COL], mode="lines",
            line=dict(width=1.4, color="#1A5FB4"), name="Prediction",
            showlegend=(i == 1), legendgroup="pred",
        ), row=1, col=i)
        if truth_col in w.columns:
            core = w[w.get("IS_CORE", pd.Series(dtype=float)) == 1]
            if len(core):
                fig.add_trace(go.Scattergl(
                    x=core[truth_col], y=core[DEPTH_COL], mode="markers",
                    marker=dict(size=6, color="#E01B24", symbol="diamond",
                                line=dict(width=1, color="white")),
                    name="Core plugs", showlegend=(i == 1), legendgroup="core",
                ), row=1, col=i)
        fig.update_xaxes(title_text=f"{TARGET_UNITS[target]}", row=1, col=i)

    if "FACIES" in w.columns:
        col_i = len(TARGETS) + 1
        fig.add_trace(go.Scattergl(
            x=w["FACIES"] + np.random.default_rng(0).uniform(-0.25, 0.25, len(w)),
            y=w[DEPTH_COL], mode="markers",
            marker=dict(size=4, color=[FACIES_COLORS.get(int(f), "#999") for f in w["FACIES"]]),
            showlegend=False,
        ), row=1, col=col_i)
        fig.update_xaxes(tickvals=list(FACIES_NAMES.keys()),
                         ticktext=["Sd", "SdSh", "ShSd", "Sh"], row=1, col=col_i)

    fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
    fig.update_layout(height=760, title=f"Predicted geomechanical properties — {well}",
                      margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=1.06))
    return fig


def crossplot_figure(df: pd.DataFrame, x: str, y: str, color: str | None = None,
                     well: str | None = None) -> go.Figure:
    """Generic scatter crossplot (used in QC tab)."""
    sub = df[df[WELL_COL] == well] if well else df
    sample = sub.sample(n=min(len(sub), 20000), random_state=1)
    fig = go.Figure(go.Scattergl(
        x=sample[x], y=sample[y], mode="markers",
        marker=dict(size=3, opacity=0.5,
                    color=sample[color] if color and color in sample else "#1A5FB4",
                    colorscale="Viridis", showscale=bool(color)),
    ))
    fig.update_layout(height=520, xaxis_title=x, yaxis_title=y,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def coverage_scatter_figure(interval_df: pd.DataFrame, target: str,
                            nominal: float) -> go.Figure:
    """Prediction vs truth with error bars for one target (calibration view)."""
    sub = interval_df[interval_df["TARGET"] == target]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=sub["PRED"], y=sub["TRUE"], mode="markers",
        error_x=dict(
            type="data", symmetric=False,
            array=sub["HI"] - sub["PRED"], arrayminus=sub["PRED"] - sub["LO"],
            color="rgba(26,95,180,0.35)"),
        marker=dict(size=5, color="#1A5FB4", opacity=0.75),
    ))
    lims = [float(np.min([sub[["TRUE", "PRED"]].min().min(),
                          sub[["TRUE", "PRED"]].max().max()]))]
    lo, hi = lims[0], lims[0]
    lo = float(min(sub["TRUE"].min(), sub["PRED"].min(), sub["LO"].min()))
    hi = float(max(sub["TRUE"].max(), sub["PRED"].max(), sub["HI"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(dash="dash", color="#888"),
                             showlegend=False))
    inside = ((sub["TRUE"] >= sub["LO"]) & (sub["TRUE"] <= sub["HI"])).mean()
    fig.update_layout(height=520, xaxis_title="Prediction", yaxis_title="Lab truth",
                      title=f"{target}: empirical coverage {inside:.1%} (nominal {nominal:.0%})",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
# Result-explanation figures (training/CV, uncertainty, ablation)
# ---------------------------------------------------------------------------
MODEL_SHORT = {"random_forest": "Random Forest", "xgboost": "XGBoost", "mlp": "MLP"}


def oof_crossplot_grid(result) -> go.Figure:
    """Blind-well crossplots: out-of-fold lab truth vs prediction for every
    model x target, with 1:1 line and R2/RMSE annotations."""
    model_keys = result.config.model_keys
    n_r, n_c = len(TARGETS), len(model_keys)
    fig = make_subplots(rows=n_r, cols=n_c,
                        subplot_titles=[f"{MODEL_SHORT.get(k, k)} · {t}"
                                        for _r, t in enumerate(TARGETS)
                                        for k in model_keys],
                        horizontal_spacing=0.07, vertical_spacing=0.09)
    for r, target in enumerate(TARGETS, start=1):
        for c, key in enumerate(model_keys, start=1):
            oof = result.cv[key].oof
            sub = oof[oof["TARGET"] == target]
            fig.add_trace(go.Scattergl(
                x=sub["TRUE"], y=sub["PRED"], mode="markers",
                marker=dict(size=5, opacity=0.65, color="#1A5FB4"),
                showlegend=False,
                name=f"{key} {target}"), row=r, col=c)
            lo = float(min(sub["TRUE"].min(), sub["PRED"].min()))
            hi = float(max(sub["TRUE"].max(), sub["PRED"].max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                     line=dict(dash="dash", color="#888"),
                                     showlegend=False), row=r, col=c)
            r2 = r2_score(sub["TRUE"], sub["PRED"])
            rmse = root_mean_squared_error(sub["TRUE"], sub["PRED"])
            fig.add_annotation(
                xref="x domain", yref="y domain", x=0.04, y=0.96,
                showarrow=False, font=dict(size=10, color="#B4491F"),
                text=f"R²={r2:.3f}<br>RMSE={rmse:.2f}", row=r, col=c)
            fig.update_xaxes(title_text="Lab truth", row=r, col=c)
            fig.update_yaxes(title_text="Prediction", row=r, col=c)
    fig.update_layout(height=330 * n_r, margin=dict(l=10, r=10, t=40, b=10),
                      title_text="Blind-well crossplots (pooled out-of-fold)")
    return fig


def per_well_metric_heatmap(result) -> go.Figure:
    """Heatmap of per-well blind R2 (rows = wells, columns = model x target)."""
    rows = []
    for key in result.config.model_keys:
        for well, sub in result.cv[key].oof.groupby("WELL"):
            for target, s2 in sub.groupby("TARGET"):
                rows.append({"WELL": well, "COL": f"{MODEL_SHORT.get(key, key)} · {target}",
                             "R2": r2_score(s2["TRUE"], s2["PRED"]) if len(s2) > 2 else np.nan})
    piv = pd.DataFrame(rows).pivot(index="WELL", columns="COL", values="R2")
    fig = px.imshow(piv, text_auto=".2f", zmin=0, zmax=1,
                    color_continuous_scale="RdBu", aspect="auto")
    fig.update_layout(height=360, title="Per-well blind R² — which wells are hard?",
                      xaxis_title="", yaxis_title="Well",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def static_dynamic_figure(data: pd.DataFrame) -> go.Figure:
    """Evidence plot: E_static vs E_dynamic on core rows (learnable conversion).

    Shows the static/dynamic ratio as a smooth function of rock competence —
    the relationship the models must learn (never a fixed factor).
    """
    core = data[data["IS_CORE"] == 1] if "IS_CORE" in data.columns else data
    sub = core.dropna(subset=["E_DYN", "E_STAT"])
    has_vsh = "VSH_TRUE" in sub.columns
    fig = go.Figure(go.Scattergl(
        x=sub["E_DYN"], y=sub["E_STAT"], mode="markers",
        marker=dict(size=5, opacity=0.6,
                    color=sub["VSH_TRUE"] if has_vsh else "#1A5FB4",
                    colorscale="Turbo", colorbar=dict(title="Vshale"),
                    showscale=has_vsh),
        name="core plugs"))
    # binned mean relationship (the function the model must learn)
    bins = pd.cut(sub["E_DYN"], bins=8)
    trend = sub.groupby(bins, observed=True).agg(x=("E_DYN", "mean"), y=("E_STAT", "mean"))
    fig.add_trace(go.Scatter(x=trend["x"], y=trend["y"], mode="lines+markers",
                             line=dict(color="#B4491F", width=3),
                             name="binned mean (target relationship)"))
    # a fixed 0.5x factor for contrast
    lim = float(sub["E_DYN"].max())
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, 0.5 * lim], mode="lines",
                             line=dict(dash="dot", color="#777"),
                             name="fixed 0.5× factor (wrong)"))
    fig.update_layout(height=480, xaxis_title="E_dynamic (GPa)",
                      yaxis_title="E_static (GPa)",
                      title="Static–dynamic relationship on paired core data",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def per_well_coverage_figure(interval_df: pd.DataFrame, target: str,
                             nominal: float) -> go.Figure:
    """Empirical interval coverage per well vs the nominal level."""
    sub = interval_df[interval_df["TARGET"] == target]
    cov = sub.groupby("WELL").apply(
        lambda s: ((s["TRUE"] >= s["LO"]) & (s["TRUE"] <= s["HI"])).mean(),
        include_groups=False).sort_values()
    fig = go.Figure(go.Bar(x=cov.index, y=cov.to_numpy(),
                           marker_color="#1A5FB4", name="coverage"))
    fig.add_hline(y=nominal, line_dash="dash", line_color="#B4491F",
                  annotation_text=f"nominal {nominal:.0%}",
                  annotation_position="top right")
    fig.update_layout(height=380, yaxis_title="Empirical coverage", yaxis_range=[0, 1.05],
                      title=f"{target}: interval coverage per well",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def ablation_delta_figure(abl: pd.DataFrame) -> go.Figure:
    """ΔR² (with Vp − without Vp) per model × target."""
    fig = px.bar(abl, x="Target", y="dR2_Vp", color="Model", barmode="group",
                 color_discrete_sequence=["#1A5FB4", "#B4491F", "#21918C"])
    fig.add_hline(y=0, line_color="#444")
    fig.update_layout(height=400, yaxis_title="ΔR² (with Vp − without Vp)",
                      title="Marginal value of the sonic log (Vp) — blind-well R²",
                      legend=dict(orientation="h", y=1.15),
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig
