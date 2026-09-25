"""
Unit and integration tests for the Breast Cancer Flask REST API Layer.

Validates:
- Health check endpoint (200 OK for ready, 503 for unhealthy)
- Model metadata endpoint
- User prediction endpoint (grouped and flat inputs, validation errors, 200/400 codes)
- Clinical prediction endpoint (full 30 cytological measurements)
- Content-type validation (requiring application/json)
- Standardized error handling and status codes (400, 404, 405, 503)
- Real inference smoke test through Flask test client
"""

import pytest
import json
import pandas as pd

from member3b.breast_cancer.src.api import create_app
from member3b.breast_cancer.src.service import BreastCancerPredictionService, DISEASE_KEY
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data


@pytest.fixture
def app():
    """Provides a configured Flask test application."""
    test_app = create_app(test_config={"TESTING": True})
    return test_app


@pytest.fixture
def client(app):
    """Provides a Flask test client for HTTP requests."""
    return app.test_client()


@pytest.fixture
def sample_flat_features():
    """Provides a valid flat 30-feature dictionary from raw dataset row 0."""
    df = load_raw_breast_cancer_data()
    cols = [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
    return df.iloc[0][cols].to_dict()


@pytest.fixture
def sample_grouped_features(sample_flat_features):
    """Provides a valid grouped 30-feature dictionary."""
    mean_feats = {k: v for k, v in sample_flat_features.items() if k.endswith("_mean")}
    se_feats = {k: v for k, v in sample_flat_features.items() if k.endswith("_se")}
    worst_feats = {k: v for k, v in sample_flat_features.items() if k.endswith("_worst")}
    return {
        "mean_features": mean_feats,
        "se_features": se_feats,
        "worst_features": worst_feats,
    }


def test_health_endpoint_healthy(client):
    """Verify GET /api/breast-cancer/health returns 200 OK when ready."""
    response = client.get("/api/breast-cancer/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ready"
    assert data["disease"] == DISEASE_KEY
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_health_endpoint_unhealthy():
    """Verify GET /api/breast-cancer/health returns 503 when service reports unhealthy."""
    class MockUnhealthyService(BreastCancerPredictionService):
        def health_check(self):
            return {
                "status": "unhealthy",
                "disease": DISEASE_KEY,
                "model_loaded": False,
                "preprocessor_loaded": False,
                "message": "Model failed to load.",
            }

    mock_app = create_app(
        test_config={"TESTING": True},
        service=MockUnhealthyService(),
    )
    mock_client = mock_app.test_client()

    response = mock_client.get("/api/breast-cancer/health")
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "unhealthy"


def test_model_info_endpoint(client):
    """Verify GET /api/breast-cancer/model-info returns 200 OK and valid metadata."""
    response = client.get("/api/breast-cancer/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["disease"] == DISEASE_KEY
    assert data["model"] == "Logistic Regression"
    assert data["threshold"] == 0.50
    assert data["feature_count"] == 30
    assert "class_mapping" in data
    assert data["class_mapping"]["0"] == "Benign" or data["class_mapping"][0] == "Benign"
    assert len(data["features"]) == 30


def test_predict_user_grouped_input(client, sample_grouped_features):
    """Verify POST /api/breast-cancer/predict/user with grouped features succeeds."""
    response = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(sample_grouped_features),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["disease"] == DISEASE_KEY
    assert data["prediction"] in ["Benign", "Malignant"]
    assert data["prediction_class"] in [0, 1]
    assert "probability" in data
    assert "risk_percentage" in data
    assert data["model"] == "Logistic Regression"
    assert data["threshold"] == 0.50


def test_predict_user_flat_input(client, sample_flat_features):
    """Verify POST /api/breast-cancer/predict/user with flat features succeeds."""
    response = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(sample_flat_features),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["prediction_class"] in [0, 1]


def test_predict_clinical_valid_input(client, sample_flat_features):
    """Verify POST /api/breast-cancer/predict/clinical succeeds with complete features."""
    response = client.post(
        "/api/breast-cancer/predict/clinical",
        data=json.dumps(sample_flat_features),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["disease"] == DISEASE_KEY
    assert data["prediction"] in ["Benign", "Malignant"]


def test_predict_user_missing_json(client):
    """Verify POST /api/breast-cancer/predict/user without JSON returns 400 Bad Request."""
    response = client.post(
        "/api/breast-cancer/predict/user",
        data="non-json raw string",
        content_type="text/plain",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "invalid_request"


def test_predict_clinical_missing_json(client):
    """Verify POST /api/breast-cancer/predict/clinical with empty body returns 400."""
    response = client.post(
        "/api/breast-cancer/predict/clinical",
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_predict_user_incomplete_features(client, sample_flat_features):
    """Verify POST /api/breast-cancer/predict/user with missing feature returns 400."""
    incomplete = sample_flat_features.copy()
    del incomplete["radius_mean"]

    response = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(incomplete),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Missing required feature" in data["message"]


def test_predict_clinical_invalid_numeric(client, sample_flat_features):
    """Verify POST /api/breast-cancer/predict/clinical with non-numeric value returns 400."""
    invalid = sample_flat_features.copy()
    invalid["texture_mean"] = "corrupted_text"

    response = client.post(
        "/api/breast-cancer/predict/clinical",
        data=json.dumps(invalid),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Invalid non-numeric value" in data["message"]


def test_predict_clinical_extra_feature(client, sample_flat_features):
    """Verify POST /api/breast-cancer/predict/clinical with extra feature returns 400."""
    extra = sample_flat_features.copy()
    extra["unexpected_feature"] = 999.9

    response = client.post(
        "/api/breast-cancer/predict/clinical",
        data=json.dumps(extra),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Unexpected feature encountered" in data["message"]


def test_user_clinical_cross_consistency(client, sample_flat_features):
    """Verify user and clinical endpoints return identical probabilities for same input."""
    res_user = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(sample_flat_features),
        content_type="application/json",
    ).get_json()

    res_clin = client.post(
        "/api/breast-cancer/predict/clinical",
        data=json.dumps(sample_flat_features),
        content_type="application/json",
    ).get_json()

    assert res_user["prediction"] == res_clin["prediction"]
    assert res_user["prediction_class"] == res_clin["prediction_class"]
    assert abs(res_user["probability"] - res_clin["probability"]) < 1e-6
    assert abs(res_user["risk_percentage"] - res_clin["risk_percentage"]) < 1e-4


def test_method_not_allowed(client):
    """Verify GET on POST-only predict endpoint returns 405."""
    response = client.get("/api/breast-cancer/predict/user")
    assert response.status_code == 405
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "method_not_allowed"


def test_not_found_endpoint(client):
    """Verify unknown URL returns 404 with standardized JSON error."""
    response = client.get("/api/breast-cancer/unknown-endpoint")
    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "not_found"


def test_real_dataset_inference_smoke_test(client):
    """Verify real inference through the Flask client on a known Malignant case (row 0)."""
    df = load_raw_breast_cancer_data()
    cols = [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
    row_0_payload = df.iloc[0][cols].to_dict()

    response = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(row_0_payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["prediction"] == "Malignant"
    assert data["prediction_class"] == 1
    assert data["probability"] == 1.0
    assert data["risk_percentage"] == 100.0
