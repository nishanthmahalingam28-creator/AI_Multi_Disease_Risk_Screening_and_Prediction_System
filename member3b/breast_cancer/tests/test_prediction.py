"""
Unit tests for the Breast Cancer Prediction Interface.

Covers:
- Model and preprocessor artifact loading
- Valid inference on 30 cytological features
- Deterministic class mappings (0 = Benign, 1 = Malignant)
- Probability range [0.0, 1.0] and risk percentage scaling
- Validation robustness (missing features, non-numeric, NaN, extra features)
- Feature order invariance (dictionary insertion order independence)
- Model consistency across repeated calls
"""

import math
import random
import numpy as np
import pandas as pd
import pytest

from member3b.breast_cancer.src.predict import (
    load_artifacts,
    predict_breast_cancer,
    predict_user,
    predict_clinical,
    validate_input,
    EXPECTED_PREDICTOR_FEATURES,
)


@pytest.fixture
def sample_valid_input():
    """Provides a realistic 30-feature input dictionary."""
    return {
        "radius_mean": 17.99,
        "texture_mean": 10.38,
        "perimeter_mean": 122.8,
        "area_mean": 1001.0,
        "smoothness_mean": 0.1184,
        "compactness_mean": 0.2776,
        "concavity_mean": 0.3001,
        "concave points_mean": 0.1471,
        "symmetry_mean": 0.2419,
        "fractal_dimension_mean": 0.07871,
        "radius_se": 1.095,
        "texture_se": 0.9053,
        "perimeter_se": 8.589,
        "area_se": 153.4,
        "smoothness_se": 0.006399,
        "compactness_se": 0.04904,
        "concavity_se": 0.05373,
        "concave points_se": 0.01587,
        "symmetry_se": 0.03003,
        "fractal_dimension_se": 0.006193,
        "radius_worst": 25.38,
        "texture_worst": 17.33,
        "perimeter_worst": 184.6,
        "area_worst": 2019.0,
        "smoothness_worst": 0.1622,
        "compactness_worst": 0.6656,
        "concavity_worst": 0.7119,
        "concave points_worst": 0.2654,
        "symmetry_worst": 0.4601,
        "fractal_dimension_worst": 0.1189,
    }


def test_1_model_loading():
    """Test 1: Verify the saved trained model loads successfully."""
    _, model, _, _ = load_artifacts(force_reload=True)
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_2_preprocessor_loading():
    """Test 2: Verify the saved preprocessor loads successfully."""
    preprocessor, _, _, _ = load_artifacts(force_reload=True)
    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")


def test_3_valid_prediction(sample_valid_input):
    """Test 3: Provide a valid 30-feature input and verify status == success."""
    result = predict_breast_cancer(sample_valid_input)
    assert result["status"] == "success"
    assert "prediction" in result
    assert "prediction_class" in result
    assert "probability" in result
    assert "risk_percentage" in result
    assert result["model"] == "Logistic Regression"
    assert result["threshold"] == 0.50


def test_4_prediction_class(sample_valid_input):
    """Test 4: Verify prediction class is either 0 or 1, and matches label."""
    result = predict_breast_cancer(sample_valid_input)
    assert result["prediction_class"] in [0, 1]
    if result["prediction_class"] == 0:
        assert result["prediction"] == "Benign"
    else:
        assert result["prediction"] == "Malignant"


def test_5_probability(sample_valid_input):
    """Test 5: Verify malignant probability is between 0.0 and 1.0."""
    result = predict_breast_cancer(sample_valid_input)
    prob = result["probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_6_risk_percentage(sample_valid_input):
    """Test 6: Verify risk_percentage == probability * 100 within tolerance."""
    result = predict_breast_cancer(sample_valid_input)
    expected_pct = result["probability"] * 100.0
    assert abs(result["risk_percentage"] - expected_pct) < 0.05


def test_7_missing_feature(sample_valid_input):
    """Test 7: Remove one required feature and verify that validation fails."""
    corrupted_input = sample_valid_input.copy()
    del corrupted_input["radius_mean"]

    result = predict_breast_cancer(corrupted_input)
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert "Missing required feature" in result["message"]
    assert "radius_mean" in result["message"]


def test_8_invalid_numeric_value(sample_valid_input):
    """Test 8: Pass a non-numeric value and verify that validation fails."""
    corrupted_input = sample_valid_input.copy()
    corrupted_input["texture_mean"] = "non_numeric_str"

    result = predict_breast_cancer(corrupted_input)
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert "Invalid non-numeric value" in result["message"]
    assert "texture_mean" in result["message"]


def test_9_nan_value(sample_valid_input):
    """Test 9: Pass NaN and verify that validation fails."""
    corrupted_input = sample_valid_input.copy()
    corrupted_input["area_mean"] = float("nan")

    result = predict_breast_cancer(corrupted_input)
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert "cannot be NaN" in result["message"]
    assert "area_mean" in result["message"]


def test_10_extra_feature(sample_valid_input):
    """Test 10: Add an unexpected feature and verify that validation fails."""
    corrupted_input = sample_valid_input.copy()
    corrupted_input["unknown_feature"] = 123.45

    result = predict_breast_cancer(corrupted_input)
    assert result["status"] == "error"
    assert result["error_type"] == "validation_error"
    assert "Unexpected feature encountered" in result["message"]
    assert "unknown_feature" in result["message"]


def test_11_feature_order_independence(sample_valid_input):
    """Test 11: Verify that different dictionary insertion order produces identical prediction."""
    # Shuffled key-value pairs
    items = list(sample_valid_input.items())
    random.Random(42).shuffle(items)
    shuffled_input = dict(items)

    result_original = predict_breast_cancer(sample_valid_input)
    result_shuffled = predict_breast_cancer(shuffled_input)

    assert result_original["prediction"] == result_shuffled["prediction"]
    assert result_original["prediction_class"] == result_shuffled["prediction_class"]
    assert abs(result_original["probability"] - result_shuffled["probability"]) < 1e-6
    assert abs(result_original["risk_percentage"] - result_shuffled["risk_percentage"]) < 1e-4


def test_12_model_consistency(sample_valid_input):
    """Test 12: Run the same valid input twice and verify consistency."""
    res1 = predict_breast_cancer(sample_valid_input)
    res2 = predict_breast_cancer(sample_valid_input)

    assert res1 == res2


def test_predict_user_and_clinical_interfaces(sample_valid_input):
    """Verify predict_user and predict_clinical wrappers work identically and reliably."""
    user_res = predict_user(sample_valid_input)
    clin_res = predict_clinical(sample_valid_input)

    assert user_res["status"] == "success"
    assert clin_res["status"] == "success"
    assert user_res["prediction"] == clin_res["prediction"]
    assert user_res["probability"] == clin_res["probability"]
