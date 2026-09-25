"""
Unit tests for Member 3B Unified Cancer Prediction REST API Layer.

Verifies:
- GET  /api/member3b/health (200 OK on ready, 503 on degraded/unavailable)
- GET  /api/member3b/model-info (200 OK on success, safe metadata)
- POST /api/member3b/predict (200 OK, envelope validation, disease routing)
- Hardened HTTP handling: malformed JSON, duplicate keys, non-JSON Content-Type
- Error handling: 400, 404, 405, 500, 503 JSON responses with zero leakages
"""

from typing import Dict, Any
from unittest.mock import MagicMock
import json
import re
import pytest

from member3b.common.api import create_app
from member3b.common.service import CancerPredictionService, UNIFIED_SERVICE_NAME
from member3b.breast_cancer.src.service import BreastCancerPredictionService
from member3b.lung_cancer.src.service import LungCancerPredictionService
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data
from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data


@pytest.fixture
def app():
    """Provides a fresh Flask application instance for testing."""
    return create_app(test_config={"TESTING": True})


@pytest.fixture
def client(app):
    """Provides a Flask HTTP test client."""
    return app.test_client()


@pytest.fixture
def valid_breast_payload() -> Dict[str, Any]:
    """Provides valid 30-feature Breast Cancer feature dictionary."""
    df = load_raw_breast_cancer_data()
    feature_cols = [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
    return {col: float(df.iloc[0][col]) for col in feature_cols}


@pytest.fixture
def valid_lung_payload() -> Dict[str, Any]:
    """Provides valid 15-feature Lung Cancer feature dictionary."""
    df = load_raw_lung_cancer_data()
    row = df.iloc[0]
    return {
        "gender": str(row["GENDER"]),
        "age": float(row["AGE"]),
        "smoking": int(row["SMOKING"]),
        "yellow_fingers": int(row["YELLOW_FINGERS"]),
        "anxiety": int(row["ANXIETY"]),
        "peer_pressure": int(row["PEER_PRESSURE"]),
        "chronic_disease": int(row["CHRONIC DISEASE"]),
        "fatigue": int(row["FATIGUE "]),
        "allergy": int(row["ALLERGY "]),
        "wheezing": int(row["WHEEZING"]),
        "alcohol_consuming": int(row["ALCOHOL CONSUMING"]),
        "coughing": int(row["COUGHING"]),
        "shortness_of_breath": int(row["SHORTNESS OF BREATH"]),
        "swallowing_difficulty": int(row["SWALLOWING DIFFICULTY"]),
        "chest_pain": int(row["CHEST PAIN"]),
    }


# =============================================================================
# 1. Health Endpoint Tests
# =============================================================================

def test_health_endpoint_healthy(client):
    """Verifies GET /api/member3b/health returns HTTP 200 when all models are ready."""
    response = client.get("/api/member3b/health")
    assert response.status_code == 200
    assert response.is_json
    assert "application/json" in response.content_type

    data = response.get_json()
    assert data["status"] == "ready"
    assert data["service"] == UNIFIED_SERVICE_NAME
    assert "breast_cancer" in data["diseases"]
    assert "lung_cancer" in data["diseases"]


def test_health_endpoint_degraded():
    """Verifies GET /api/member3b/health returns HTTP 503 if any sub-service is degraded."""
    mock_service = MagicMock(spec=CancerPredictionService)
    mock_service.health_check.return_value = {
        "status": "unavailable",
        "service": UNIFIED_SERVICE_NAME,
        "diseases": {
            "breast_cancer": {"status": "unhealthy"},
            "lung_cancer": {"status": "ready"},
        },
    }
    custom_app = create_app(test_config={"TESTING": True}, service=mock_service)
    resp = custom_app.test_client().get("/api/member3b/health")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "unavailable"


# =============================================================================
# 2. Model Information Endpoint Tests
# =============================================================================

def test_model_info_endpoint_success(client):
    """Verifies GET /api/member3b/model-info returns HTTP 200 and safe metadata."""
    response = client.get("/api/member3b/model-info")
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["status"] == "success"
    assert data["service"] == UNIFIED_SERVICE_NAME
    assert "breast_cancer" in data["models"]
    assert "lung_cancer" in data["models"]

    # Verify absence of leaked secrets or internal paths
    text = response.get_data(as_text=True)
    assert not re.search(r"[A-Za-z]:\\", text)
    assert "joblib" not in text
    assert "coef_" not in text


def test_model_info_endpoint_failure():
    """Verifies GET /api/member3b/model-info returns HTTP 503 on service failure."""
    mock_service = MagicMock(spec=CancerPredictionService)
    mock_service.get_model_info.return_value = {
        "status": "unavailable",
        "service": UNIFIED_SERVICE_NAME,
        "error_type": "service_error",
        "message": "Unified cancer prediction service is currently unavailable.",
    }
    custom_app = create_app(test_config={"TESTING": True}, service=mock_service)
    resp = custom_app.test_client().get("/api/member3b/model-info")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "unavailable"


# =============================================================================
# 3. Unified Prediction Endpoint: Valid Invocations
# =============================================================================

def test_predict_endpoint_breast_cancer(client, valid_breast_payload):
    """Verifies POST /api/member3b/predict with breast_cancer payload returns HTTP 200."""
    body = {
        "disease": "breast_cancer",
        "data": valid_breast_payload,
    }
    response = client.post("/api/member3b/predict", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data["status"] == "success"
    assert data["disease"] == "breast_cancer"
    assert "prediction" in data
    assert "probability" in data


def test_predict_endpoint_lung_cancer(client, valid_lung_payload):
    """Verifies POST /api/member3b/predict with lung_cancer payload returns HTTP 200."""
    body = {
        "disease": "lung_cancer",
        "data": valid_lung_payload,
    }
    response = client.post("/api/member3b/predict", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data["status"] == "success"
    assert data["disease"] == "lung_cancer"
    assert "prediction" in data
    assert "probability" in data


# =============================================================================
# 4. Envelope Validation Tests
# =============================================================================

def test_predict_endpoint_missing_disease(client, valid_lung_payload):
    """Verifies missing 'disease' field returns HTTP 400 validation error."""
    body = {"data": valid_lung_payload}
    response = client.post("/api/member3b/predict", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error_type"] == "validation_error"
    assert "disease" in response.get_json()["message"]


def test_predict_endpoint_missing_data(client):
    """Verifies missing 'data' field returns HTTP 400 validation error."""
    body = {"disease": "lung_cancer"}
    response = client.post("/api/member3b/predict", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error_type"] == "validation_error"
    assert "data" in response.get_json()["message"]


def test_predict_endpoint_unsupported_disease(client, valid_lung_payload):
    """Verifies unmodeled or invalid disease name returns HTTP 400."""
    body = {"disease": "stroke", "data": valid_lung_payload}
    response = client.post("/api/member3b/predict", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error_type"] == "validation_error"
    assert "Unsupported disease" in response.get_json()["message"]


def test_predict_endpoint_invalid_data_container(client):
    """Verifies non-dict 'data' container returns HTTP 400."""
    for invalid_data in ["not_a_dict", [1, 2, 3], 1234, True, {}]:
        body = {"disease": "lung_cancer", "data": invalid_data}
        response = client.post("/api/member3b/predict", data=json.dumps(body), content_type="application/json")
        assert response.status_code == 400
        assert response.get_json()["error_type"] == "validation_error"


# =============================================================================
# 5. Hardened HTTP Behavior Tests
# =============================================================================

def test_predict_endpoint_empty_body(client):
    """Verifies empty POST request returns HTTP 400."""
    response = client.post("/api/member3b/predict", data="", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error_type"] == "invalid_request"


def test_predict_endpoint_malformed_json(client):
    """Verifies malformed JSON payload returns HTTP 400 with safe error message."""
    response = client.post("/api/member3b/predict", data="{unclosed: json", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error_type"] == "invalid_request"
    assert "Traceback" not in response.get_data(as_text=True)


def test_predict_endpoint_duplicate_keys(client):
    """Verifies duplicate keys in JSON payload return HTTP 400."""
    payload_with_dups = '{"disease": "lung_cancer", "disease": "breast_cancer", "data": {}}'
    response = client.post("/api/member3b/predict", data=payload_with_dups, content_type="application/json")
    assert response.status_code == 400
    assert "Duplicate JSON field" in response.get_json()["message"]


def test_predict_endpoint_non_json_content_type(client, valid_lung_payload):
    """Verifies non-JSON Content-Type returns HTTP 400."""
    body = json.dumps({"disease": "lung_cancer", "data": valid_lung_payload})
    response = client.post("/api/member3b/predict", data=body, content_type="text/plain")
    assert response.status_code == 400
    assert "Content-Type application/json" in response.get_json()["message"]


def test_standard_error_handlers(client):
    """Verifies 400, 404, 405 error handlers return structured JSON."""
    # 404 Not Found
    r404 = client.get("/api/member3b/nonexistent")
    assert r404.status_code == 404
    assert r404.is_json
    assert r404.get_json()["error_type"] == "not_found"

    # 405 Method Not Allowed
    r405 = client.get("/api/member3b/predict")
    assert r405.status_code == 405
    assert r405.is_json
    assert r405.get_json()["error_type"] == "method_not_allowed"


def test_service_error_maps_to_503():
    """Verifies internal service unavailable error results in HTTP 503."""
    mock_service = MagicMock(spec=CancerPredictionService)
    mock_service.predict.return_value = {
        "status": "error",
        "service": UNIFIED_SERVICE_NAME,
        "error_type": "service_error",
        "message": "Disease prediction service is currently unavailable.",
    }
    custom_app = create_app(test_config={"TESTING": True}, service=mock_service)
    body = {"disease": "lung_cancer", "data": {"age": 50}}
    resp = custom_app.test_client().post("/api/member3b/predict", data=json.dumps(body), content_type="application/json")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "error"
