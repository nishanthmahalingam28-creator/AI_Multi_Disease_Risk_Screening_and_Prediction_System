"""
Integration test suite for Breast Cancer Prediction Module.

Validates end-to-end integration across the architectural stack:
Flask REST API -> BreastCancerPredictionService -> predict.py -> StandardScaler -> Logistic Regression Model.

Focuses specifically on:
1. Service and preprocessor/model health check
2. Model metadata information endpoint
3. User prediction workflow (grouped and flat formats)
4. Clinical prediction workflow
5. User vs Clinical cross-endpoint consistency
6. Rejection of incomplete feature vectors (missing feature)
7. Rejection of unauthorized feature vectors (unexpected feature)
8. Rejection of non-numeric feature values
9. Rejection of null/None feature values
10. Model and preprocessor artifact readiness and caching
11. Rejection of NaN and infinite numeric values
12. Detection and rejection of duplicate feature keys in JSON payloads
"""

import pytest
import json
import math
import pandas as pd

from member3b.breast_cancer.src.api import create_app
from member3b.breast_cancer.src.service import BreastCancerPredictionService, DISEASE_KEY
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data


@pytest.fixture
def app():
    """Provides a fresh Flask application instance for integration testing."""
    return create_app(test_config={"TESTING": True})


@pytest.fixture
def client(app):
    """Provides an HTTP test client."""
    return app.test_client()


@pytest.fixture
def service():
    """Provides a live BreastCancerPredictionService instance."""
    return BreastCancerPredictionService()


@pytest.fixture
def valid_sample_features():
    """Provides a valid 30-feature dictionary extracted from raw dataset row 0."""
    df = load_raw_breast_cancer_data()
    feature_cols = [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
    return {col: float(df.iloc[0][col]) for col in feature_cols}


@pytest.fixture
def valid_grouped_sample_features(valid_sample_features):
    """Provides the valid 30-feature vector partitioned into mean, se, and worst groups."""
    return {
        "mean_features": {k: v for k, v in valid_sample_features.items() if k.endswith("_mean")},
        "se_features": {k: v for k, v in valid_sample_features.items() if k.endswith("_se")},
        "worst_features": {k: v for k, v in valid_sample_features.items() if k.endswith("_worst")},
    }


# =============================================================================
# 1. Service Health Check Integration
# =============================================================================
def test_integration_service_health(client, service):
    """Verifies service health check via both Python service and REST API."""
    # Direct service capability verification
    health = service.health_check()
    assert health["status"] == "ready"
    assert health["disease"] == DISEASE_KEY
    assert health["model_loaded"] is True
    assert health["preprocessor_loaded"] is True

    # REST endpoint verification
    response = client.get("/api/breast-cancer/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ready"
    assert data["disease"] == DISEASE_KEY
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


# =============================================================================
# 2. Model Metadata Information Integration
# =============================================================================
def test_integration_model_info_endpoint(client, service):
    """Verifies that model metadata exposes safe information without internal secrets."""
    # Direct service
    info = service.get_model_info()
    assert info["disease"] == DISEASE_KEY
    assert info["model"] == "Logistic Regression"
    assert info["threshold"] == 0.50
    assert info["feature_count"] == 30
    assert len(info["features"]) == 30

    # API endpoint
    response = client.get("/api/breast-cancer/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["disease"] == DISEASE_KEY
    assert data["model"] == "Logistic Regression"
    assert data["threshold"] == 0.50
    assert data["feature_count"] == 30

    # Verify no leaked internals
    for forbidden in ["coefficients", "intercept", "filepath", "joblib", "stack", "env"]:
        assert forbidden not in data


# =============================================================================
# 3. Valid User Prediction Integration
# =============================================================================
def test_integration_valid_user_prediction(client, valid_grouped_sample_features, valid_sample_features):
    """Verifies successful user risk screening for both grouped and flat feature vectors."""
    # Test grouped structure
    resp_grouped = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(valid_grouped_sample_features),
        content_type="application/json",
    )
    assert resp_grouped.status_code == 200
    data_g = resp_grouped.get_json()
    assert data_g["status"] == "success"
    assert data_g["disease"] == DISEASE_KEY
    assert data_g["prediction"] in ["Benign", "Malignant"]
    assert data_g["prediction_class"] in [0, 1]
    assert isinstance(data_g["probability"], (float, int))
    assert 0.0 <= data_g["probability"] <= 1.0
    assert data_g["model"] == "Logistic Regression"
    assert data_g["threshold"] == 0.50

    # Test flat structure
    resp_flat = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(valid_sample_features),
        content_type="application/json",
    )
    assert resp_flat.status_code == 200
    data_f = resp_flat.get_json()
    assert data_f["status"] == "success"
    assert data_f["prediction"] == data_g["prediction"]
    assert data_f["probability"] == data_g["probability"]


# =============================================================================
# 4. Valid Clinical Prediction Integration
# =============================================================================
def test_integration_valid_clinical_prediction(client, valid_sample_features):
    """Verifies successful clinical diagnostic prediction through Flask REST API."""
    response = client.post(
        "/api/breast-cancer/predict/clinical",
        data=json.dumps(valid_sample_features),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["disease"] == DISEASE_KEY
    assert data["prediction"] in ["Benign", "Malignant"]
    assert data["prediction_class"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model"] == "Logistic Regression"
    assert data["threshold"] == 0.50


# =============================================================================
# 5. User vs Clinical Prediction Consistency
# =============================================================================
def test_integration_user_clinical_consistency(client, service, valid_sample_features):
    """
    Verifies that identical input vectors dispatched through User and Clinical
    modes yield mathematically identical prediction labels, probabilities, and risks.
    """
    # Service layer consistency
    user_svc_res = service.predict_user(valid_sample_features)
    clin_svc_res = service.predict_clinical(valid_sample_features)
    assert user_svc_res["prediction"] == clin_svc_res["prediction"]
    assert user_svc_res["prediction_class"] == clin_svc_res["prediction_class"]
    assert user_svc_res["probability"] == clin_svc_res["probability"]
    assert user_svc_res["risk_percentage"] == clin_svc_res["risk_percentage"]

    # HTTP API layer consistency
    resp_user = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(valid_sample_features),
        content_type="application/json",
    )
    resp_clin = client.post(
        "/api/breast-cancer/predict/clinical",
        data=json.dumps(valid_sample_features),
        content_type="application/json",
    )
    assert resp_user.status_code == 200
    assert resp_clin.status_code == 200
    data_u = resp_user.get_json()
    data_c = resp_clin.get_json()

    assert data_u["prediction"] == data_c["prediction"]
    assert data_u["prediction_class"] == data_c["prediction_class"]
    assert data_u["probability"] == data_c["probability"]
    assert data_u["risk_percentage"] == data_c["risk_percentage"]


# =============================================================================
# 6. Incomplete Input Rejection (Missing Feature)
# =============================================================================
def test_integration_missing_feature_rejected(client, valid_sample_features):
    """Verifies that omission of any required feature returns HTTP 400 with descriptive error."""
    incomplete = dict(valid_sample_features)
    del incomplete["radius_mean"]

    for endpoint in ["/api/breast-cancer/predict/user", "/api/breast-cancer/predict/clinical"]:
        resp = client.post(
            endpoint,
            data=json.dumps(incomplete),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "validation_error"
        assert "radius_mean" in data["message"]


# =============================================================================
# 7. Unauthorized Input Rejection (Unexpected Feature)
# =============================================================================
def test_integration_unexpected_feature_rejected(client, valid_sample_features):
    """Verifies that payloads with unmodeled features return HTTP 400."""
    extra = dict(valid_sample_features)
    extra["tumor_stage"] = "Stage II"

    for endpoint in ["/api/breast-cancer/predict/user", "/api/breast-cancer/predict/clinical"]:
        resp = client.post(
            endpoint,
            data=json.dumps(extra),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "validation_error"
        assert "Unexpected feature encountered" in data["message"]


# =============================================================================
# 8. Non-Numeric Input Rejection
# =============================================================================
def test_integration_non_numeric_feature_rejected(client, valid_sample_features):
    """Verifies that non-numeric strings return HTTP 400 validation error."""
    corrupted = dict(valid_sample_features)
    corrupted["perimeter_mean"] = "corrupted_text"

    for endpoint in ["/api/breast-cancer/predict/user", "/api/breast-cancer/predict/clinical"]:
        resp = client.post(
            endpoint,
            data=json.dumps(corrupted),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "validation_error"
        assert "Invalid non-numeric value" in data["message"]


# =============================================================================
# 9. Null Value Rejection
# =============================================================================
def test_integration_null_feature_rejected(client, valid_sample_features):
    """Verifies that null/None values are rejected with HTTP 400."""
    null_payload = dict(valid_sample_features)
    null_payload["texture_mean"] = None

    for endpoint in ["/api/breast-cancer/predict/user", "/api/breast-cancer/predict/clinical"]:
        resp = client.post(
            endpoint,
            data=json.dumps(null_payload),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "validation_error"
        assert "cannot be None" in data["message"]


# =============================================================================
# 10. Model & Preprocessor Availability
# =============================================================================
def test_integration_model_preprocessor_availability(service):
    """Verifies model and preprocessor are available, loaded, and can transform data."""
    service._ensure_artifacts()
    assert service._model is not None
    assert service._preprocessor is not None
    assert hasattr(service._model, "predict")
    assert hasattr(service._preprocessor, "transform")


# =============================================================================
# 11. NaN and Infinity Input Rejection
# =============================================================================
def test_integration_nan_and_infinity_rejected(client, valid_sample_features):
    """Verifies that NaN and Infinite numeric inputs are rejected with HTTP 400."""
    # Test NaN string representation
    nan_payload = dict(valid_sample_features)
    nan_payload["area_mean"] = "NaN"
    resp_nan = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(nan_payload),
        content_type="application/json",
    )
    assert resp_nan.status_code == 400
    assert "cannot be NaN" in resp_nan.get_json()["message"]

    # Test Infinity string representation
    inf_payload = dict(valid_sample_features)
    inf_payload["area_mean"] = "Infinity"
    resp_inf = client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(inf_payload),
        content_type="application/json",
    )
    assert resp_inf.status_code == 400
    assert "cannot be infinite" in resp_inf.get_json()["message"]


# =============================================================================
# 12. Duplicate JSON Key Rejection
# =============================================================================
def test_integration_duplicate_feature_keys_rejected(client, valid_sample_features):
    """Verifies that raw JSON with duplicate keys is detected and rejected with HTTP 400."""
    # Form raw JSON with duplicate "radius_mean"
    json_str = json.dumps(valid_sample_features)
    # Inject duplicate key at the start
    duplicate_json = '{"radius_mean": 99.9, ' + json_str[1:]

    resp = client.post(
        "/api/breast-cancer/predict/user",
        data=duplicate_json,
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert "Duplicate feature key encountered" in data["message"]
