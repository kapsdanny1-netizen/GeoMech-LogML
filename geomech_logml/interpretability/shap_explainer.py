"""SHAP interpretability: summary (beeswarm), dependence and per-prediction plots.

* Tree models (Random Forest, XGBoost) use ``shap.TreeExplainer`` — exact and fast.
* The MLP uses ``shap.KernelExplainer`` over a k-means background summary
  (model-agnostic, slower — the UI subsamples and shows progress).

Every prediction can be explained individually (waterfall) in addition to the
global summary + dependence views, satisfying the "full SHAP for every
prediction" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shap

from geomech_logml.config import GLOBAL_SEED
from geomech_logml.models.registry import MODEL_SPECS

__all__ = ["ShapExplainer", "top_features_by_shap"]


def _unwrap_pipeline(model: Any) -> tuple[Any, Any]:
    """Return (predict_function, base_model) for a bare estimator or sklearn Pipeline."""
    if hasattr(model, "steps"):  # sklearn Pipeline (MLP case)
        scaler = model.named_steps["scaler"]
        mlp = model.named_steps["mlp"]
        columns = getattr(scaler, "feature_names_in_", None)

        def predict_fn(X):
            if isinstance(X, np.ndarray) and columns is not None:
                X = pd.DataFrame(X, columns=columns)
            return mlp.predict(scaler.transform(X))

        return predict_fn, mlp
    return (lambda X: model.predict(X)), model


@dataclass
class ShapExplainer:
    """Builds and caches a SHAP explainer for one (model_key, target) pair."""

    model_key: str
    target: str
    model: Any
    background: pd.DataFrame
    seed: int = GLOBAL_SEED

    def __post_init__(self) -> None:
        self.kind = MODEL_SPECS[self.model_key].shap_kind
        self._predict_fn, self._base = _unwrap_pipeline(self.model)
        if self.kind == "tree":
            self.explainer = shap.TreeExplainer(self._base)
            self.expected_value = float(np.asarray(self.explainer.expected_value).ravel()[0])
        else:  # kernel
            summary = shap.kmeans(self.background.to_numpy(), k=min(25, len(self.background)))
            self.explainer = shap.KernelExplainer(self._predict_fn, summary)
            self.expected_value = float(
                np.mean(self._predict_fn(self.background.iloc[:200])))

    # ------------------------------------------------------------------
    def shap_values(self, X: pd.DataFrame, nsamples: int = 200) -> np.ndarray:
        """SHAP matrix (n_rows, n_features) for X."""
        if self.kind == "tree":
            vals = self.explainer.shap_values(X, check_additivity=False)
        else:
            vals = self.explainer.shap_values(
                X.to_numpy(),
                nsamples=nsamples,
                silent=True,
            )
        vals = np.asarray(vals)
        if vals.ndim == 3:            # (n, features, outputs) in some versions
            vals = vals[..., 0]
        return vals

    # ------------------------------------------------------------------
    # Figures (returned as matplotlib Figure objects; caller displays/closes)
    # ------------------------------------------------------------------
    def summary_figure(self, X: pd.DataFrame, max_display: int = 12) -> plt.Figure:
        """Beeswarm summary plot for one target."""
        vals = self.shap_values(X)
        shap.summary_plot(vals, X, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.suptitle(f"SHAP summary — {MODEL_SPECS[self.model_key].label} · {self.target}",
                     y=1.02, fontsize=11)
        return fig

    def dependence_figure(self, X: pd.DataFrame, feature: str) -> plt.Figure:
        """SHAP dependence plot for one feature (auto-interaction colouring)."""
        vals = self.shap_values(X)
        shap.dependence_plot(feature, vals, X, show=False)
        fig = plt.gcf()
        fig.suptitle(f"{self.target} — SHAP dependence: {feature}", y=1.02, fontsize=11)
        return fig

    def waterfall_figure(self, x_row: pd.Series, nsamples: int = 200) -> plt.Figure:
        """Per-prediction explanation (waterfall) for a single row."""
        Xrow = x_row.to_frame().T
        vals = self.shap_values(Xrow, nsamples=nsamples)[0]
        base = self.expected_value
        pred = float(self._predict_fn(Xrow)[0])
        explanation = shap.Explanation(
            values=vals,
            base_values=base,                      # scalar — required by waterfall
            data=Xrow.to_numpy()[0],
            feature_names=list(Xrow.columns),
        )
        shap.plots.waterfall(explanation, max_display=12, show=False)
        fig = plt.gcf()
        fig.suptitle(f"{self.target} — prediction {pred:.3f}", y=1.02, fontsize=11)
        return fig

    def top_features(self, X: pd.DataFrame, k: int = 4) -> list[str]:
        """Feature names ranked by mean |SHAP| (descending)."""
        vals = self.shap_values(X)
        means = np.abs(vals).mean(axis=0)
        order = np.argsort(means)[::-1]
        return [str(X.columns[i]) for i in order[:k]]

    # ------------------------------------------------------------------
    def mean_abs_shap_figure(self, X: pd.DataFrame, max_display: int = 12) -> "plt.Figure":
        """Horizontal bar chart of mean |SHAP| per feature (global importance)."""
        vals = self.shap_values(X)
        means = pd.Series(np.abs(vals).mean(axis=0), index=X.columns)
        means = means.sort_values(ascending=True).tail(max_display)
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(means) + 1.4))
        ax.barh(means.index, means.to_numpy(), color="#1A5FB4")
        ax.set_xlabel(f"mean |SHAP value| — {self.target}")
        ax.set_title(f"Global feature importance ({MODEL_SPECS[self.model_key].label})",
                     fontsize=11)
        for i, v in enumerate(means.to_numpy()):
            ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8, color="#333")
        fig.tight_layout()
        return fig


def top_features_by_shap(shap_matrix: np.ndarray, feature_names: list[str], k: int = 4) -> list[str]:
    """Utility: rank feature names by mean |SHAP| from a precomputed matrix."""
    means = np.abs(np.asarray(shap_matrix)).mean(axis=0)
    order = np.argsort(means)[::-1]
    return [feature_names[i] for i in order[:k]]
