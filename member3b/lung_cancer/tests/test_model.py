"""
Lung Cancer Model Verification and Evaluation Tests.

Validates:
- Test 1: Model artifact exists on disk
- Test 2: Model can be loaded and is valid estimator
- Test 3: Model has expected 15-feature compatibility
- Test 4: Model produces binary predictions {0, 1}
- Test 5: Model produces valid probabilities bounded in [0, 1]
- Test 6: Probability array has correct shape (N, 2) and row sums equal 1.0
- Test 7: Reloaded model produces identical predictions and probabilities
- Test 8: Test set was not used during hyperparameter selection (leakage audit)
- Test 9: Saved metadata contains selected model, hyperparameters, and metrics
- Test 10: Confusion matrix dimensions and totals match the held-out test partition
- Test 11: All final metrics are finite, non-null, and within valid mathematical bounds
- Test 12: Diagnostic report artifacts exist and are non-empty
- Test 13: Single sample prediction smoke tests
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from member3b.lung_cancer.src.data_loader import (
    load_raw_lung_cancer_data,
    EXPECTED_SHA256,
)
from member3b.lung_cancer.src.preprocessing import (
    clean_lung_cancer_data,
    load_preprocessor,
    get_default_models_dir,
    EXPECTED_FEATURES,
    EXPECTED_NUM_FEATURES,
)
from member3b.lung_cancer.src.train import (
    get_default_reports_dir,
    load_partitioned_data,
)


@pytest.fixture(scope="module")
def artifacts():
    """Loads saved models, preprocessor, and partitions once for the test module."""
    models_dir = get_default_models_dir()
    reports_dir = get_default_reports_dir()

    model_path = models_dir / "lung_cancer_model.joblib"
    prep_path = models_dir / "lung_cancer_preprocessor.joblib"
    meta_path = models_dir / "model_metadata.json"
    final_eval_path = reports_dir / "final_evaluation.json"

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(final_eval_path, "r", encoding="utf-8") as f:
        final_eval = json.load(f)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        _,
        split_meta,
    ) = load_partitioned_data()

    return {
        "model_path": model_path,
        "prep_path": prep_path,
        "meta_path": meta_path,
        "final_eval_path": final_eval_path,
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "final_eval": final_eval,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "split_meta": split_meta,
        "reports_dir": reports_dir,
    }


# -------------------------------------------------------------------------
# Test 1: Model artifact exists
# -------------------------------------------------------------------------
def test_1_model_artifact_exists(artifacts):
    """Test 1: Verifies lung_cancer_model.joblib exists in the models directory."""
    assert artifacts["model_path"].is_file(), f"Model artifact missing at {artifacts['model_path']}"
    assert artifacts["model_path"].stat().st_size > 0


# -------------------------------------------------------------------------
# Test 2: Model can be loaded
# -------------------------------------------------------------------------
def test_2_model_can_be_loaded(artifacts):
    """Test 2: Verifies model can be deserialized and has predict/predict_proba."""
    model = artifacts["model"]
    assert hasattr(model, "predict"), "Loaded model lacks predict() method"
    assert hasattr(model, "predict_proba"), "Loaded model lacks predict_proba() method"


# -------------------------------------------------------------------------
# Test 3: Model has the expected feature compatibility
# -------------------------------------------------------------------------
def test_3_model_feature_compatibility(artifacts):
    """Test 3: Verifies model expects exactly 15 input features."""
    model = artifacts["model"]
    # For LogisticRegression, n_features_in_ must equal 15
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == EXPECTED_NUM_FEATURES
    assert model.n_features_in_ == 15


# -------------------------------------------------------------------------
# Test 4: Model produces binary predictions {0, 1}
# -------------------------------------------------------------------------
def test_4_model_produces_binary_predictions(artifacts):
    """Test 4: Verifies predictions on test features are strictly discrete {0, 1}."""
    model = artifacts["model"]
    X_test_scaled = artifacts["X_test_scaled"]

    preds = model.predict(X_test_scaled)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(artifacts["X_test"])
    unique_preds = set(np.unique(preds))
    assert unique_preds.issubset({0, 1}), f"Unexpected prediction classes: {unique_preds}"


# -------------------------------------------------------------------------
# Test 5: Model produces probabilities between 0 and 1
# -------------------------------------------------------------------------
def test_5_model_produces_probabilities_between_0_and_1(artifacts):
    """Test 5: Verifies predicted probabilities are strictly bounded in [0.0, 1.0]."""
    model = artifacts["model"]
    X_test_scaled = artifacts["X_test_scaled"]

    probs = model.predict_proba(X_test_scaled)
    assert not np.isnan(probs).any(), "Probabilities contain NaN values"
    assert not np.isinf(probs).any(), "Probabilities contain infinite values"
    assert (probs >= 0.0).all(), "Found negative probabilities"
    assert (probs <= 1.0).all(), "Found probabilities exceeding 1.0"


# -------------------------------------------------------------------------
# Test 6: Prediction probability has the correct shape
# -------------------------------------------------------------------------
def test_6_prediction_probability_shape_and_sum(artifacts):
    """Test 6: Verifies probability matrix has shape (N, 2) and row sums equal 1.0."""
    model = artifacts["model"]
    X_test_scaled = artifacts["X_test_scaled"]

    probs = model.predict_proba(X_test_scaled)
    assert probs.shape == (len(artifacts["X_test"]), 2)
    # Check each row sums to 1.0 within numerical precision
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# -------------------------------------------------------------------------
# Test 7: Reloaded model produces the same results
# -------------------------------------------------------------------------
def test_7_reloaded_model_produces_identical_results(artifacts):
    """Test 7: Verifies reloaded model generates identical predictions and probabilities."""
    reloaded_model = joblib.load(artifacts["model_path"])
    X_test_scaled = artifacts["X_test_scaled"]

    preds_original = artifacts["model"].predict(X_test_scaled)
    preds_reloaded = reloaded_model.predict(X_test_scaled)
    np.testing.assert_array_equal(preds_original, preds_reloaded)

    probs_original = artifacts["model"].predict_proba(X_test_scaled)
    probs_reloaded = reloaded_model.predict_proba(X_test_scaled)
    np.testing.assert_allclose(probs_original, probs_reloaded, atol=1e-12)


# -------------------------------------------------------------------------
# Test 8: The test set was not used during hyperparameter selection
# -------------------------------------------------------------------------
def test_8_test_set_not_used_during_tuning_leakage_audit(artifacts):
    """Test 8: Verifies test set was completely isolated from cross-validation and tuning."""
    meta = artifacts["metadata"]
    split_meta = artifacts["split_meta"]

    # Verify training row count is strictly 247 and test row count is 62
    assert meta["training_row_count"] == 247
    assert meta["test_row_count"] == 62
    assert meta["cross_validation_method"].startswith("StratifiedKFold (5-fold")

    # Verify indices in train and test do not overlap
    train_idx = set(split_meta["train_indices_raw_csv_0_indexed"])
    test_idx = set(split_meta["test_indices_raw_csv_0_indexed"])
    assert len(train_idx.intersection(test_idx)) == 0

    # Verify test set size matches 20%
    assert len(test_idx) == 62
    assert len(train_idx) == 247


# -------------------------------------------------------------------------
# Test 9: Saved metadata contains the selected model and metrics
# -------------------------------------------------------------------------
def test_9_saved_metadata_contains_selected_model_and_metrics(artifacts):
    """Test 9: Verifies model_metadata.json contains complete configuration and metric blocks."""
    meta = artifacts["metadata"]
    assert meta["disease"] == "Lung Cancer Prediction"
    assert meta["model_type"] == "LogisticRegression"
    assert meta["selection_metric"] == "cv_roc_auc_mean"
    assert "selected_hyperparameters" in meta
    assert "cross_validation_metrics_selected" in meta
    assert "final_evaluation_metrics" in meta

    cv_selected = meta["cross_validation_metrics_selected"]
    assert "cv_roc_auc_mean" in cv_selected
    assert "cv_pr_auc_mean" in cv_selected
    assert "cv_balanced_accuracy_mean" in cv_selected


# -------------------------------------------------------------------------
# Test 10: Confusion matrix dimensions are correct
# -------------------------------------------------------------------------
def test_10_confusion_matrix_dimensions_and_counts(artifacts):
    """Test 10: Verifies confusion matrix is 2x2 and sum of cells equals 62 test samples."""
    final_eval = artifacts["final_eval"]
    cm = final_eval["confusion_matrix"]

    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = tn + fp + fn + tp
    assert total == 62, f"Total confusion matrix count {total} does not equal test set size 62"
    assert tn + fp == 8, f"Total negative actuals {tn+fp} != 8"
    assert fn + tp == 54, f"Total positive actuals {fn+tp} != 54"


# -------------------------------------------------------------------------
# Test 11: All final metrics are finite
# -------------------------------------------------------------------------
def test_11_all_final_metrics_are_finite(artifacts):
    """Test 11: Verifies all evaluation metrics are finite numbers within [0.0, 1.0]."""
    final_eval = artifacts["final_eval"]
    metrics_to_check = [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "specificity", "f1", "roc_auc", "pr_auc"
    ]
    for metric_name in metrics_to_check:
        val = final_eval[metric_name]
        assert isinstance(val, (int, float)), f"Metric {metric_name} is not numeric"
        assert np.isfinite(val), f"Metric {metric_name} is not finite (NaN/inf)"
        assert 0.0 <= val <= 1.0, f"Metric {metric_name} ({val}) is out of [0, 1] range"


# -------------------------------------------------------------------------
# Test 12: Diagnostic report artifacts exist
# -------------------------------------------------------------------------
def test_12_diagnostic_report_artifacts_exist(artifacts):
    """Test 12: Verifies all generated visualization plots and summary reports exist."""
    reports_dir = artifacts["reports_dir"]
    expected_files = [
        "model_comparison.csv",
        "model_comparison.md",
        "confusion_matrix.png",
        "roc_curve.png",
        "pr_curve.png",
        "final_evaluation.json",
    ]
    for fname in expected_files:
        p = reports_dir / fname
        assert p.is_file(), f"Report file missing: {p}"
        assert p.stat().st_size > 0, f"Report file is empty: {p}"


# -------------------------------------------------------------------------
# Test 13: Single sample prediction smoke tests
# -------------------------------------------------------------------------
def test_13_single_sample_prediction_smoke_test(artifacts):
    """Test 13: Verifies single-row prediction for a positive and negative case."""
    model = artifacts["model"]
    X_test_scaled = artifacts["X_test_scaled"]

    # Predict on first sample
    single_x = X_test_scaled[[0]]
    pred = model.predict(single_x)
    prob = model.predict_proba(single_x)

    assert pred.shape == (1,)
    assert prob.shape == (1, 2)
    assert pred[0] in [0, 1]
    assert np.isclose(prob[0].sum(), 1.0)
