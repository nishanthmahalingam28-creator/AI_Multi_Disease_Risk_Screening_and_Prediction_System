"""
Unit tests for the Breast Cancer Prediction Service Layer.

Validates:
- Service initialization and dependency injection
- Health check functionality
- Model metadata retrieval
- User and clinical prediction workflows
- Validation error propagation
- Consistency between repeated and cross-mode invocations
- Mathematical and class mapping correctness
"""

import pytest
import pandas as pd

from member3b.breast_cancer.src.service import BreastCancerPredictionService, DISEASE_KEY
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data


@pytest.fixture
def service():
    """Provides an initialized BreastCancerPredictionService instance."""
    return BreastCancerPredictionService()


@pytest.fixture
def valid_sample_input():
    """Provides a valid 30-feature dictionary extracted from raw data."""
    df = load_raw_breast_cancer_data()
    features = [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
    return df.iloc[0][features].to_dict()


def test_1_service_creation(service):
    """Test 1: Verify BreastCancerPredictionService instantiates cleanly."""
    assert service is not None
    assert isinstance(service, BreastCancerPredictionService)


def test_2_health_check(service):
    """Test 2: Verify health_check reports status == 'ready' and artifacts loaded."""
    health = service.health_check()
    assert health["status"] == "ready"
    assert health["disease"] == DISEASE_KEY
    assert health["model_loaded"] is True
    assert health["preprocessor_loaded"] is True


def test_3_model_information(service):
    """Test 3: Verify get_model_info returns expected model metadata."""
    info = service.get_model_info()
    assert info["disease"] == DISEASE_KEY
    assert info["model"] == "Logistic Regression"
    assert info["threshold"] == 0.50
    assert info["feature_count"] == 30
    assert info["class_mapping"] == {0: "Benign", 1: "Malignant"}
    assert len(info["features"]) == 30


def test_4_user_prediction(service, valid_sample_input):
    """Test 4: Verify predict_user succeeds with valid input."""
    res = service.predict_user(valid_sample_input)
    assert res["status"] == "success"
    assert res["disease"] == DISEASE_KEY
    assert res["prediction"] in ["Benign", "Malignant"]
    assert res["prediction_class"] in [0, 1]
    assert "probability" in res
    assert "risk_percentage" in res
    assert res["model"] == "Logistic Regression"
    assert res["threshold"] == 0.50


def test_5_clinical_prediction(service, valid_sample_input):
    """Test 5: Verify predict_clinical succeeds with valid input."""
    res = service.predict_clinical(valid_sample_input)
    assert res["status"] == "success"
    assert res["disease"] == DISEASE_KEY
    assert res["prediction"] in ["Benign", "Malignant"]
    assert res["prediction_class"] in [0, 1]
    assert "probability" in res
    assert "risk_percentage" in res


def test_6_user_validation_error(service, valid_sample_input):
    """Test 6: Verify incomplete input returns structured error in predict_user."""
    incomplete_input = valid_sample_input.copy()
    del incomplete_input["radius_mean"]

    res = service.predict_user(incomplete_input)
    assert res["status"] == "error"
    assert res["disease"] == DISEASE_KEY
    assert res["error_type"] == "validation_error"
    assert "Missing required feature" in res["message"]


def test_7_clinical_validation_error(service, valid_sample_input):
    """Test 7: Verify non-numeric input returns structured error in predict_clinical."""
    invalid_input = valid_sample_input.copy()
    invalid_input["texture_mean"] = "corrupted_text"

    res = service.predict_clinical(invalid_input)
    assert res["status"] == "error"
    assert res["disease"] == DISEASE_KEY
    assert res["error_type"] == "validation_error"
    assert "Invalid non-numeric value" in res["message"]


def test_8_consistent_result(service, valid_sample_input):
    """Test 8: Verify repeated calls with identical input yield identical results."""
    res1 = service.predict_user(valid_sample_input)
    res2 = service.predict_user(valid_sample_input)
    assert res1 == res2


def test_9_prediction_delegation(service, valid_sample_input, monkeypatch):
    """Test 9: Verify service delegates to predict_user/predict_clinical rather than a second model."""
    called = {"user": False}

    import member3b.breast_cancer.src.service as s_mod

    original_predict_user = s_mod._predict_user_core

    def mock_predict_user(input_data, **kwargs):
        called["user"] = True
        return original_predict_user(input_data, **kwargs)

    monkeypatch.setattr(s_mod, "_predict_user_core", mock_predict_user)

    res = service.predict_user(valid_sample_input)
    assert called["user"] is True
    assert res["status"] == "success"


def test_10_probability_validity(service, valid_sample_input):
    """Test 10: Verify probability is strictly within [0.0, 1.0]."""
    res = service.predict_clinical(valid_sample_input)
    prob = res["probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_11_risk_percentage_validity(service, valid_sample_input):
    """Test 11: Verify risk_percentage is strictly within [0.0, 100.0]."""
    res = service.predict_clinical(valid_sample_input)
    risk_pct = res["risk_percentage"]
    assert isinstance(risk_pct, (int, float))
    assert 0.0 <= risk_pct <= 100.0
    assert abs(risk_pct - (res["probability"] * 100.0)) < 0.05


def test_12_class_mapping(service):
    """Test 12: Verify class mapping is 0 = Benign, 1 = Malignant."""
    info = service.get_model_info()
    mapping = info["class_mapping"]
    assert mapping[0] == "Benign"
    assert mapping[1] == "Malignant"


def test_13_user_clinical_consistency(service, valid_sample_input):
    """Test 13: Verify predict_user and predict_clinical yield identical model outputs."""
    user_res = service.predict_user(valid_sample_input)
    clin_res = service.predict_clinical(valid_sample_input)

    assert user_res["status"] == clin_res["status"]
    assert user_res["prediction"] == clin_res["prediction"]
    assert user_res["prediction_class"] == clin_res["prediction_class"]
    assert abs(user_res["probability"] - clin_res["probability"]) < 1e-6
    assert abs(user_res["risk_percentage"] - clin_res["risk_percentage"]) < 1e-4
