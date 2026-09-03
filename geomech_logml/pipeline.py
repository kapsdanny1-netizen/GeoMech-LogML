"""End-to-end experiment orchestration.

``run_experiment`` glues together the full workflow:

    raw logs → clean → engineer features → well-wise CV per model
    → honest uncertainty validation (nested conformal + out-of-well QRF)
    → final models trained on all core rows → full-curve predictions + intervals

``run_ablation`` re-runs the pooled metrics with and without Vp-derived features
(the key methodological ablation: does the optional sonic log measurably help?).

Everything is deterministic given ``ExperimentConfig.seed``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from geomech_logml.config import (
    CORE_FLAG,
    GLOBAL_SEED,
    TARGETS,
    TARGET_BOUNDS,
    WELL_COL,
    DEPTH_COL,
)
from geomech_logml.models.evaluate import CVResult, metrics_table, run_well_wise_cv
from geomech_logml.models.registry import (
    DEFAULT_MODELS,
    MODEL_SPECS,
    build_model,
)
from geomech_logml.preprocessing.cv import make_splitter
from geomech_logml.preprocessing.features import (
    DEFAULT_FEATURE_SET,
    clean_logs,
    engineer_features,
    get_feature_names,
    split_xy_core,
)
from geomech_logml.uncertainty.conformal import (
    conformal_quantile,
    empirical_coverage,
    well_wise_conformal_intervals,
)
from geomech_logml.uncertainty.qrf import QuantileForest

__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment", "run_ablation"]


# ---------------------------------------------------------------------------
# Config / result containers
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """All knobs for one experiment run (fully serialisable)."""
    feature_set: str = DEFAULT_FEATURE_SET
    model_keys: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    cv_strategy: str = "well_kfold"          # "well_kfold" | "leave_one_well_out"
    n_splits: int = 5
    alpha: float = 0.10                      # interval miscoverage rate (90% PI)
    seed: int = GLOBAL_SEED
    hyper_overrides: dict[str, dict] = field(default_factory=dict)
    validate_uncertainty: bool = True        # nested conformal + out-of-well QRF
    uncertainty_model: str = "random_forest"  # model family used for honest PI validation


@dataclass
class ExperimentResult:
    """Everything the UI / reports need after one experiment."""
    config: ExperimentConfig
    data: pd.DataFrame                      # cleaned + engineered full log frame
    X_core: pd.DataFrame
    Y_core: pd.DataFrame
    groups: pd.Series
    feature_names: list[str]
    cv: dict[str, CVResult]                 # model_key -> CVResult
    metrics: pd.DataFrame                   # pooled OOF metrics (model × target)
    honest_conformal: pd.DataFrame | None   # nested well-wise conformal rows
    qrf_oof: pd.DataFrame | None            # out-of-well QRF interval rows
    curves: pd.DataFrame                    # full-curve point predictions + conformal PI
    final_models: dict[str, dict]           # model_key -> {target: estimator}
    runtime_seconds: float
    curves_user: pd.DataFrame | None = None  # transfer mode: predictions on a
                                             # separate (e.g. uploaded) log frame
    data_user: pd.DataFrame | None = None    # transfer mode: engineered frame of
                                             # the uploaded logs (for plots/PDF)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_experiment(df_raw: pd.DataFrame, cfg: ExperimentConfig | None = None,
                   predict_on: pd.DataFrame | None = None) -> ExperimentResult:
    """Run the complete train/evaluate/predict cycle on a canonical log frame.

    Parameters
    ----------
    df_raw : training log frame (must contain core-calibrated rows).
    cfg : experiment configuration.
    predict_on : optional *separate* raw log frame to predict after training
        (transfer mode: e.g. train on synthetic/cored data, apply to uploaded
        legacy wells). Results land in ``ExperimentResult.curves_user``.
    """
    cfg = cfg or ExperimentConfig()
    t0 = time.perf_counter()

    data = engineer_features(clean_logs(df_raw))
    X, Y, groups = split_xy_core(data, cfg.feature_set)
    if groups.nunique() < 3:
        raise ValueError(f"Need >= 3 cored wells for well-wise CV; got {groups.nunique()}.")
    feats = get_feature_names(data, cfg.feature_set)

    splitter = make_splitter(cfg.cv_strategy, n_wells=groups.nunique(),
                             n_splits=cfg.n_splits, seed=cfg.seed)

    cv_results: dict[str, CVResult] = {}
    for key in cfg.model_keys:
        cv_results[key] = run_well_wise_cv(
            X, Y, groups, key, splitter=splitter, seed=cfg.seed,
            overrides=cfg.hyper_overrides.get(key),
        )
    table = metrics_table(cv_results)

    # ---- honest interval validation (uncertainty provider model) --------------
    honest_conformal: pd.DataFrame | None = None
    qrf_oof: pd.DataFrame | None = None
    if cfg.validate_uncertainty:
        ukey = cfg.uncertainty_model
        if ukey not in cfg.model_keys:
            cv_results[ukey] = run_well_wise_cv(
                X, Y, groups, ukey, splitter=splitter, seed=cfg.seed,
                overrides=cfg.hyper_overrides.get(ukey))
            table = metrics_table(cv_results)

        def _fit_predict(Xtr: pd.DataFrame, ytr: pd.Series, Xev: pd.DataFrame):
            est = MODEL_SPECS[ukey].build(seed=cfg.seed, overrides=cfg.hyper_overrides.get(ukey))
            est.fit(Xtr, np.asarray(ytr))
            return est.predict(Xev)

        honest_conformal = well_wise_conformal_intervals(
            _fit_predict, X, Y, groups, alpha=cfg.alpha)

        # Out-of-well QRF: fit per CV fold, intervals on the held-out well.
        # Interval forests use larger leaves than the point model so leaf pools are
        # wide enough for stable quantiles (Meinshausen, 2006 recommends min leaf
        # size ~ 10 for quantile estimation).
        rows = []
        for fold, (tr, te) in enumerate(splitter.split(X, groups=groups.to_numpy())):
            for target in TARGETS:
                forest = build_model("random_forest", seed=cfg.seed,
                                     overrides={"min_samples_leaf": 10})
                forest.fit(X.iloc[tr].to_numpy(), Y.iloc[tr][target].to_numpy())
                qrf = QuantileForest(forest, X.iloc[tr].to_numpy(),
                                     Y.iloc[tr][target].to_numpy())
                lo, med, hi = qrf.predict_intervals(X.iloc[te].to_numpy(), alpha=cfg.alpha)
                truth = Y.iloc[te][target].to_numpy()
                for k, j in enumerate(te):
                    rows.append({"WELL": groups.iloc[j], "FOLD": fold, "TARGET": target,
                                 "TRUE": float(truth[k]), "PRED": float(med[k]),
                                 "LO": float(lo[k]), "HI": float(hi[k]),
                                 "WIDTH": float(hi[k] - lo[k])})
        qrf_oof = pd.DataFrame(rows)

    # ---- final models + full-curve predictions --------------------------------
    final_models = {k: res.final_models for k, res in cv_results.items()}
    curves = predict_curves(data, feats, final_models, cv_results, alpha=cfg.alpha)

    curves_user = None
    data_user = None
    if predict_on is not None:
        data_user = engineer_features(clean_logs(predict_on))
        curves_user = predict_curves(data_user, feats, final_models, cv_results,
                                     alpha=cfg.alpha)

    return ExperimentResult(
        config=cfg, data=data, X_core=X, Y_core=Y, groups=groups, feature_names=feats,
        cv=cv_results, metrics=table,
        honest_conformal=honest_conformal, qrf_oof=qrf_oof,
        curves=curves, final_models=final_models,
        runtime_seconds=time.perf_counter() - t0,
        curves_user=curves_user, data_user=data_user,
    )


def predict_curves(
    data: pd.DataFrame,
    feature_names: list[str],
    final_models: dict[str, dict],
    cv_results: dict[str, CVResult],
    alpha: float = 0.10,
    pred_prefixes: bool = True,
) -> pd.DataFrame:
    """Predict every cleaned log row with every model; attach conformal PI bands.

    Point predictions are clipped to physically admissible target bounds.
    Interval half-widths come from each model's pooled well-wise OOF residuals.
    """
    mask = data[feature_names].notna().all(axis=1)
    sub = data.loc[mask]
    Xc = sub[feature_names].astype(float)

    id_cols = [WELL_COL, DEPTH_COL] + ([CORE_FLAG] if CORE_FLAG in sub.columns else [])
    out = sub[id_cols].copy()
    for truth_col in TARGETS:
        if truth_col in sub.columns:
            out[f"{truth_col}_TRUE"] = sub[truth_col].to_numpy()
    for facies_col in ("FACIES", "VSH_TRUE"):
        if facies_col in sub.columns:
            out[facies_col] = sub[facies_col].to_numpy()

    for key, bundle in final_models.items():
        for target in TARGETS:
            pred = bundle[target].predict(Xc)
            lo_b, hi_b = TARGET_BOUNDS[target]
            pred = np.clip(pred, lo_b, hi_b)
            if pred_prefixes:
                out[f"{target}_{key}"] = pred
            # conformal band from this model's OOF residuals
            oof = cv_results[key].oof
            oof_t = oof[oof["TARGET"] == target]
            width = conformal_quantile((oof_t["TRUE"] - oof_t["PRED"]).abs().to_numpy(), alpha)
            out[f"{target}_{key}_LO"] = pred - width
            out[f"{target}_{key}_HI"] = pred + width
    return out.reset_index(drop=True)


def qrf_curve_intervals(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_curve: pd.DataFrame,
    alpha: float = 0.10,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """Row-wise QRF bands for full curves (per-target interval forests on core rows).

    Uses min_samples_leaf=10 (wider leaf pools) per Meinshausen's recommendation.
    NOTE: QRF bands reflect training-response spread and per-row difficulty; they do
    NOT absorb inter-well bias — the conformal bands (attached by `predict_curves`)
    are the primary calibrated intervals.
    """
    out = pd.DataFrame(index=X_curve.index)
    Xtr = X_train.to_numpy()
    for target in TARGETS:
        forest = build_model("random_forest", seed=seed,
                             overrides={"min_samples_leaf": 10})
        forest.fit(Xtr, Y_train[target].to_numpy())
        qrf = QuantileForest(forest, Xtr, Y_train[target].to_numpy())
        lo, med, hi = qrf.predict_intervals(X_curve.to_numpy(), alpha=alpha)
        out[f"{target}_QLO"] = lo
        out[f"{target}_QMED"] = med
        out[f"{target}_QHI"] = hi
    return out


# ---------------------------------------------------------------------------
# Ablation: with vs without Vp
# ---------------------------------------------------------------------------
def run_ablation(
    df_raw: pd.DataFrame,
    cfg: ExperimentConfig | None = None,
    with_vp_set: str | None = None,
    without_vp_set: str | None = None,
) -> pd.DataFrame:
    """Compare pooled OOF metrics for feature sets with vs without Vp.

    Returns a tidy table: Model × Target × {R2_with, R2_without, dR2, RMSE_with,
    RMSE_without}. Positive dR2 quantifies the marginal value of the sonic log.
    """
    cfg = cfg or ExperimentConfig()
    with_vp_set = with_vp_set or "eng_with_vp"
    without_vp_set = without_vp_set or "eng_no_vp"

    rows = []
    for fs in (with_vp_set, without_vp_set):
        sub_cfg = ExperimentConfig(
            feature_set=fs, model_keys=cfg.model_keys, cv_strategy=cfg.cv_strategy,
            n_splits=cfg.n_splits, alpha=cfg.alpha, seed=cfg.seed,
            hyper_overrides=cfg.hyper_overrides, validate_uncertainty=False)
        res = run_experiment(df_raw, sub_cfg)
        for _, r in res.metrics.iterrows():
            rows.append({"FeatureSet": fs, "HasVp": fs == with_vp_set,
                         "Model": r["Model"], "ModelKey": r["ModelKey"],
                         "Target": r["Target"], "R2": r["R2"], "RMSE": r["RMSE"]})
    long = pd.DataFrame(rows)

    wide = long.pivot_table(index=["ModelKey", "Model", "Target"], columns="HasVp",
                            values=["R2", "RMSE"])
    wide.columns = [f"{m}_{'with' if v else 'without'}_Vp" for m, v in wide.columns]
    wide = wide.reset_index()
    wide["dR2_Vp"] = wide["R2_with_Vp"] - wide["R2_without_Vp"]
    wide["dRMSE_Vp"] = wide["RMSE_without_Vp"] - wide["RMSE_with_Vp"]
    return wide


def interval_coverage_summary(res: ExperimentResult) -> pd.DataFrame:
    """Coverage/width summary for both interval methods (honest, out-of-well)."""
    rows = []
    nominal = 1.0 - res.config.alpha
    if res.honest_conformal is not None:
        for t, sub in res.honest_conformal.groupby("TARGET"):
            rows.append({"Method": "Conformal (nested well-wise)", "Target": t,
                         "Nominal": nominal,
                         "Coverage": empirical_coverage(sub["TRUE"], sub["LO"], sub["HI"]),
                         "MeanWidth": sub["WIDTH"].mean()})
    if res.qrf_oof is not None:
        for t, sub in res.qrf_oof.groupby("TARGET"):
            rows.append({"Method": "QRF (out-of-well)", "Target": t,
                         "Nominal": nominal,
                         "Coverage": empirical_coverage(sub["TRUE"], sub["LO"], sub["HI"]),
                         "MeanWidth": sub["WIDTH"].mean()})
    return pd.DataFrame(rows)


def config_as_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)
