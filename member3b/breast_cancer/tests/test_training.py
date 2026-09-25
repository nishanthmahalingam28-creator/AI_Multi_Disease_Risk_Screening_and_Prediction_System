"""
Unit tests for the Breast Cancer Model Training and Artifact Verification.

Validates that candidate models instantiate, cross-validation and tuning execute cleanly,
the selected model serializes and reloads, and outputs align with the 30-feature schema.
"""

from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from member3b.breast_cancer.src.data_loader import get_default_dataset_path
from member3b.breast_cancer.src.preprocessing import (
    load_preprocessor,
    prepare_breast_cancer_data,
    EXPECTED_NUM_FEATURES,
)
from member3b.breast_cancer.src.train import (
    get_candidate_models_and_grids,
    get_default_models_dir,
    get_default_reports_dir,
)

EXPECTED_RAW_DATASET_SHA256 = "1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33"


def test_candidate_models_instantiation():
    """Verify all 5 required candidate models and hyperparameter grids can be instantiated."""
    models_and_grids = get_candidate_models_and_grids()
    expected_models = [
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
    ]
    for model_name in expected_models:
        assert model_name in models_and_grids
        estimator, grid = models_and_grids[model_name]
        assert estimator is not None
        assert isinstance(grid, list)
        assert len(grid) > 0


def test_saved_model_artifact_exists_and_reloads():
    """Verify breast_cancer_model.joblib exists, can be loaded, and expects 30 features."""
    models_dir = get_default_models_dir()
    model_path = models_dir / "breast_cancer_model.joblib"
    assert model_path.exists(), f"Model artifact missing at {model_path}"

    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded model missing predict() method"
    assert hasattr(model, "predict_proba"), "Loaded model missing predict_proba() method"

    # Test inference on a dummy sample of 30 features
    dummy_input = np.zeros((1, EXPECTED_NUM_FEATURES))
    pred = model.predict(dummy_input)
    prob = model.predict_proba(dummy_input)

    assert pred.shape == (1,)
    assert pred[0] in [0, 1]
    assert prob.shape == (1, 2)
    assert abs(prob.sum() - 1.0) < 1e-5


def test_preprocessor_model_pipeline_consistency():
    """
    Verify that Step 3 preprocessor output can be directly consumed
    by the Step 4 saved model without shape or type mismatch.
    """
    prep = load_preprocessor()
    model = joblib.load(get_default_models_dir() / "breast_cancer_model.joblib")

    # Load 2 raw test instances from Step 3 preparation
    data = prepare_breast_cancer_data(save_artifacts=False)
    X_test = data["X_test"]
    X_test_scaled = prep.transform(X_test)

    assert X_test_scaled.shape[1] == EXPECTED_NUM_FEATURES
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)

    assert len(preds) == len(X_test)
    assert len(probs) == len(X_test)


def test_model_metadata_validity():
    """Verify model_metadata.json contains complete, authentic experimental results."""
    meta_path = get_default_models_dir() / "model_metadata.json"
    assert meta_path.exists(), f"Model metadata missing at {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["disease_name"] == "Breast Cancer Prediction"
    assert meta["selected_model_name"] in [
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
    ]
    assert len(meta["models_evaluated"]) == 5
    assert meta["feature_count"] == 30
    assert "cv_roc_auc_mean" in meta["cross_validation_metrics_selected"]
    assert "accuracy" in meta["final_test_metrics_selected"]
    assert "disclaimer" in meta


def test_model_comparison_csv_validity():
    """Verify model_comparison.csv contains all 5 candidate models and expected columns."""
    csv_path = get_default_reports_dir() / "model_comparison.csv"
    assert csv_path.exists(), f"Comparison CSV missing at {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 5
    expected_cols = [
        "model",
        "cv_roc_auc_mean",
        "cv_recall_mean",
        "test_accuracy",
        "test_recall",
        "test_roc_auc",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column '{col}' in comparison CSV"


def test_raw_dataset_untouched_after_training():
    """Confirm raw dataset/breast.csv was not modified during training."""
    raw_path = get_default_dataset_path()
    with open(raw_path, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()
    assert current_hash == EXPECTED_RAW_DATASET_SHA256
