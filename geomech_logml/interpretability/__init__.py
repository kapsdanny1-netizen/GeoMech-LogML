"""Interpretability: SHAP wrappers for every model family."""

from geomech_logml.interpretability.shap_explainer import ShapExplainer, top_features_by_shap

__all__ = ["ShapExplainer", "top_features_by_shap"]
