"""Model persistence: bundle export/import round-trip + prediction-only mode."""

from __future__ import annotations

import io

import numpy as np

import pytest

from geomech_logml.config import TARGETS, WELL_COL
from geomech_logml.models.persistence import (
    bundle_from_bytes,
    bundle_from_result,
    bundle_to_bytes,
    predict_with_bundle,
    result_from_bundle,
)
from geomech_logml.pipeline import ExperimentConfig, run_experiment


@pytest.fixture(scope="module")
def small_result(engineered):
    cfg = ExperimentConfig(feature_set="eng_with_vp",
                           model_keys=["random_forest"], n_splits=3, seed=0,
                           validate_uncertainty=False)
    return run_experiment(engineered, cfg)


def test_bundle_roundtrip(small_result):
    bundle = bundle_from_result(small_result, "random_forest")
    raw = bundle_to_bytes(bundle)
    assert isinstance(raw, bytes) and len(raw) > 1000
    back = bundle_from_bytes(io.BytesIO(raw))
    assert back["model_key"] == "random_forest"
    assert back["feature_names"] == bundle["feature_names"]
    assert set(back["models"]) == set(TARGETS)
    # estimator survives serialisation and still predicts
    pred = back["models"]["UCS"].predict(small_result.X_core.iloc[:5])
    assert np.isfinite(pred).all()


def test_bundle_rejects_foreign_files():
    with pytest.raises(ValueError):
        bundle_from_bytes(io.BytesIO(b"not a bundle at all"))


def test_predict_with_bundle_matches_direct(small_result, engineered):
    """Bundle predictions must equal the experiment's final-model predictions."""
    bundle = bundle_from_bytes(io.BytesIO(bundle_to_bytes(
        bundle_from_result(small_result, "random_forest"))))
    curves, data = predict_with_bundle(engineered, bundle)
    key = bundle["model_key"]
    joined = curves.merge(
        small_result.curves[["WELL", "DEPT", f"UCS_{key}"]], on=["WELL", "DEPT"])
    assert len(joined) == len(curves)
    np.testing.assert_allclose(joined[f"UCS_{key}_x"], joined[f"UCS_{key}_y"], rtol=1e-9)
    # intervals present and ordered
    assert (curves[f"UCS_{key}_LO"] <= curves[f"UCS_{key}"]).all()


def test_result_from_bundle_is_prediction_only(small_result, engineered):
    bundle = bundle_from_bytes(io.BytesIO(bundle_to_bytes(
        bundle_from_result(small_result, "random_forest"))))
    res = result_from_bundle(engineered, bundle)
    assert res.metrics.empty and res.cv == {}
    assert res.honest_conformal is None and res.qrf_oof is None
    assert len(res.curves) > 0
    assert len(res.X_core) > 0                      # SHAP background available
    assert f"UCS_{bundle['model_key']}" in res.curves.columns


def test_predict_with_bundle_missing_features(small_result, engineered):
    bundle = bundle_from_bytes(io.BytesIO(bundle_to_bytes(
        bundle_from_result(small_result, "random_forest"))))
    stripped = engineered.drop(columns=["VP", "PHIS", "AI"])
    with pytest.raises(ValueError, match="cannot provide"):
        predict_with_bundle(stripped, bundle)


# ---------------------------------------------------------------------------
# Transfer mode (train here, predict there)
# ---------------------------------------------------------------------------
def test_transfer_mode_predictions(small_df, engineered):
    """Uploads without core targets get predicted by models trained elsewhere."""
    cfg = ExperimentConfig(feature_set="eng_with_vp",
                           model_keys=["random_forest"], n_splits=3, seed=0,
                           validate_uncertainty=False)
    # pretend the "upload" is a different synthetic well set without any targets
    from geomech_logml.data.synthetic import SyntheticConfig, generate_well
    upload = generate_well("UPLOAD-1", SyntheticConfig(n_wells=1, seed=99, step_m=4.0))
    upload = upload.drop(columns=TARGETS + ["IS_CORE", "E_DYN", "NU_DYN"])

    res = run_experiment(small_df, cfg, predict_on=upload)
    assert res.curves_user is not None and res.data_user is not None
    assert res.curves_user[WELL_COL].unique().tolist() == ["UPLOAD-1"]
    assert "UCS_random_forest" in res.curves_user.columns
    assert res.curves_user["UCS_random_forest"].notna().all()
    assert (res.curves_user["UCS_random_forest_LO"]
            <= res.curves_user["UCS_random_forest"]).all()
    # transfer curves carry no truth columns
    assert "UCS_TRUE" not in res.curves_user.columns
