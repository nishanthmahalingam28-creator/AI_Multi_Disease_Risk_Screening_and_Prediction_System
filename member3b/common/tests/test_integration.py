"""
Integration test suite for Member 3B Unified Cancer Prediction Module.

Validates end-to-end integration across the architectural stack:
Flask REST API -> CancerPredictionService -> (Breast / Lung) Service -> ML Models.

Verifies:
- Complete prediction workflow consistency between Direct Disease APIs and Unified API
- Exact disease-specific validation preservation through Unified API
- Request isolation across alternating disease sequences
- Deterministic output across repeated predictions
- Safe failure handling and error isolation
- App factory import safety
"""

from typing import Dict, Any
from pathlib import Path
from unittest.mock import MagicMock
import json
import re
import pytest

from member3b.common.api import create_app
from member3b.common.service import CancerPredictionService, UNIFIED_SERVICE_NAME
import member3b.breast_cancer.src.api as breast_api_mod
import member3b.lung_cancer.src.api as lung_api_mod
from member3b.breast_cancer.src.service import BreastCancerPredictionService
from member3b.lung_cancer.src.service import LungCancerPredictionService
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data
from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data


@pytest.fixture
def unified_client():
    """Provides Flask test client for Unified Member 3B API."""
    app = create_app(test_config={"TESTING": True})
    return app.test_client()


@pytest.fixture
def breast_direct_client():
    """Provides Flask test client for direct Breast Cancer API."""
    app = breast_api_mod.create_app(test_config={"TESTING": True})
    return app.test_client()


@pytest.fixture
def lung_direct_client():
    """Provides Flask test client for direct Lung Cancer API."""
    app = lung_api_mod.create_app(test_config={"TESTING": True})
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
# PHASE 11: Result Consistency Between Direct APIs and Unified API
# =============================================================================

def test_integration_breast_cancer_consistency(unified_client, breast_direct_client, valid_breast_payload):
    """
    Verifies that Direct Breast API and Unified API yield mathematically identical
    predictions, classes, probabilities, risk percentages, thresholds, and models.
    """
    # 1. Direct Breast API call
    resp_direct = breast_direct_client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(valid_breast_payload),
        content_type="application/json",
    )
    assert resp_direct.status_code == 200
    direct_data = resp_direct.get_json()

    # 2. Unified Member 3B API call
    unified_envelope = {
        "disease": "breast_cancer",
        "data": valid_breast_payload,
    }
    resp_unified = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps(unified_envelope),
        content_type="application/json",
    )
    assert resp_unified.status_code == 200
    unified_data = resp_unified.get_json()

    # Compare semantic prediction values
    assert unified_data["prediction"] == direct_data["prediction"]
    assert unified_data["prediction_class"] == direct_data["prediction_class"]
    assert abs(unified_data["probability"] - direct_data["probability"]) < 1e-6
    assert abs(unified_data["risk_percentage"] - direct_data["risk_percentage"]) < 1e-4
    assert unified_data["model"] == direct_data["model"]
    assert abs(unified_data["threshold"] - direct_data["threshold"]) < 1e-6


def test_integration_lung_cancer_consistency(unified_client, lung_direct_client, valid_lung_payload):
    """
    Verifies that Direct Lung API and Unified API yield mathematically identical
    predictions, classes, probabilities, risk percentages, interpretations, and models.
    """
    # 1. Direct Lung API call
    resp_direct = lung_direct_client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_lung_payload),
        content_type="application/json",
    )
    assert resp_direct.status_code == 200
    direct_data = resp_direct.get_json()

    # 2. Unified Member 3B API call
    unified_envelope = {
        "disease": "lung_cancer",
        "data": valid_lung_payload,
    }
    resp_unified = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps(unified_envelope),
        content_type="application/json",
    )
    assert resp_unified.status_code == 200
    unified_data = resp_unified.get_json()

    # Compare semantic prediction values
    assert unified_data["prediction"] == direct_data["prediction"]
    assert unified_data["prediction_class"] == direct_data["prediction_class"]
    assert abs(unified_data["probability"] - direct_data["probability"]) < 1e-6
    assert abs(unified_data["risk_percentage"] - direct_data["risk_percentage"]) < 1e-4
    assert unified_data["model"] == direct_data["model"]
    assert abs(unified_data["threshold"] - direct_data["threshold"]) < 1e-6
    assert unified_data["screening_interpretation"] == direct_data["screening_interpretation"]


# =============================================================================
# PHASE 13: Disease-Specific Validation Preserved Through Unified API
# =============================================================================

def test_integration_breast_validation_missing_feature(unified_client, valid_breast_payload):
    """Verifies that missing feature in breast payload returns HTTP 400 validation error."""
    bad_data = dict(valid_breast_payload)
    del bad_data["radius_mean"]
    resp = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "breast_cancer", "data": bad_data}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "radius_mean" in data["message"]


def test_integration_breast_validation_unexpected_feature(unified_client, valid_breast_payload):
    """Verifies that unexpected feature in breast payload returns HTTP 400 validation error."""
    bad_data = dict(valid_breast_payload, tumor_stage="Stage II")
    resp = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "breast_cancer", "data": bad_data}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Unexpected feature" in data["message"]


def test_integration_breast_validation_nan_infinity(unified_client, valid_breast_payload):
    """Verifies that NaN or Infinity in breast payload returns HTTP 400."""
    bad_data = dict(valid_breast_payload, texture_mean=float("nan"))
    resp = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "breast_cancer", "data": bad_data}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "cannot be NaN" in resp.get_json()["message"]


def test_integration_lung_validation_missing_feature(unified_client, valid_lung_payload):
    """Verifies that missing feature in lung payload returns HTTP 400 validation error."""
    bad_data = dict(valid_lung_payload)
    del bad_data["smoking"]
    resp = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "lung_cancer", "data": bad_data}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "smoking" in data["message"]


def test_integration_lung_validation_invalid_gender(unified_client, valid_lung_payload):
    """Verifies that invalid gender in lung payload returns HTTP 400."""
    bad_data = dict(valid_lung_payload, gender="UNKNOWN")
    resp = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "lung_cancer", "data": bad_data}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "Invalid value for gender" in resp.get_json()["message"]


def test_integration_lung_validation_invalid_age(unified_client, valid_lung_payload):
    """Verifies that out-of-bound age in lung payload returns HTTP 400."""
    bad_data = dict(valid_lung_payload, age=-5)
    resp = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "lung_cancer", "data": bad_data}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "Age must be positive" in resp.get_json()["message"]


def test_integration_lung_validation_nan_infinity(unified_client, valid_lung_payload):
    """Verifies that NaN or Infinity in lung payload returns HTTP 400."""
    bad_data = dict(valid_lung_payload, age=float("nan"))
    resp = unified_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "lung_cancer", "data": bad_data}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "cannot be NaN" in resp.get_json()["message"]


# =============================================================================
# PHASE 15 & 16: Health and Model Info Failure Simulation
# =============================================================================

def test_integration_health_breast_failure_simulation():
    """Verifies unified health reports 503 and clearly identifies breast failure."""
    mock_b = MagicMock(spec=BreastCancerPredictionService)
    mock_b.health_check.return_value = {"status": "unhealthy", "disease": "breast_cancer"}
    mock_l = MagicMock(spec=LungCancerPredictionService)
    mock_l.health_check.return_value = {"status": "ready", "disease": "lung_cancer"}

    svc = CancerPredictionService(breast_service=mock_b, lung_service=mock_l)
    app = create_app(test_config={"TESTING": True}, service=svc)
    client = app.test_client()

    resp = client.get("/api/member3b/health")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["diseases"]["breast_cancer"]["status"] == "unhealthy"
    assert data["diseases"]["lung_cancer"]["status"] == "ready"


def test_integration_health_lung_failure_simulation():
    """Verifies unified health reports 503 and clearly identifies lung failure."""
    mock_b = MagicMock(spec=BreastCancerPredictionService)
    mock_b.health_check.return_value = {"status": "ready", "disease": "breast_cancer"}
    mock_l = MagicMock(spec=LungCancerPredictionService)
    mock_l.health_check.return_value = {"status": "unavailable", "disease": "lung_cancer"}

    svc = CancerPredictionService(breast_service=mock_b, lung_service=mock_l)
    app = create_app(test_config={"TESTING": True}, service=svc)
    client = app.test_client()

    resp = client.get("/api/member3b/health")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["diseases"]["breast_cancer"]["status"] == "ready"
    assert data["diseases"]["lung_cancer"]["status"] == "unavailable"


# =============================================================================
# PHASE 17: Request State Isolation
# =============================================================================

def test_integration_request_isolation_alternating(unified_client, valid_breast_payload, valid_lung_payload):
    """Verifies interleaved calls across diseases produce isolated, uncorrupted predictions."""
    b_env = {"disease": "breast_cancer", "data": valid_breast_payload}
    l_env = {"disease": "lung_cancer", "data": valid_lung_payload}

    # Sequence: Breast -> Lung -> Breast
    rb1 = unified_client.post("/api/member3b/predict", data=json.dumps(b_env), content_type="application/json")
    assert rb1.status_code == 200
    assert rb1.get_json()["disease"] == "breast_cancer"

    rl = unified_client.post("/api/member3b/predict", data=json.dumps(l_env), content_type="application/json")
    assert rl.status_code == 200
    assert rl.get_json()["disease"] == "lung_cancer"

    rb2 = unified_client.post("/api/member3b/predict", data=json.dumps(b_env), content_type="application/json")
    assert rb2.status_code == 200
    assert rb2.get_json()["prediction"] == rb1.get_json()["prediction"]

    # Sequence: Valid -> Invalid -> Valid
    bad_env = {"disease": "lung_cancer", "data": {"age": -99}}
    r_bad = unified_client.post("/api/member3b/predict", data=json.dumps(bad_env), content_type="application/json")
    assert r_bad.status_code == 400

    rb3 = unified_client.post("/api/member3b/predict", data=json.dumps(b_env), content_type="application/json")
    assert rb3.status_code == 200
    assert rb3.get_json()["prediction"] == rb1.get_json()["prediction"]


# =============================================================================
# PHASE 18: Repeated Prediction Consistency
# =============================================================================

def test_integration_repeated_breast_predictions(unified_client, valid_breast_payload):
    """Verifies 10 repeated Breast Cancer calls yield identical deterministic results."""
    env = {"disease": "breast_cancer", "data": valid_breast_payload}
    first_res = None
    for _ in range(10):
        resp = unified_client.post("/api/member3b/predict", data=json.dumps(env), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        if first_res is None:
            first_res = data
        else:
            assert data["prediction"] == first_res["prediction"]
            assert abs(data["probability"] - first_res["probability"]) < 1e-9


def test_integration_repeated_lung_predictions(unified_client, valid_lung_payload):
    """Verifies 10 repeated Lung Cancer calls yield identical deterministic results."""
    env = {"disease": "lung_cancer", "data": valid_lung_payload}
    first_res = None
    for _ in range(10):
        resp = unified_client.post("/api/member3b/predict", data=json.dumps(env), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        if first_res is None:
            first_res = data
        else:
            assert data["prediction"] == first_res["prediction"]
            assert abs(data["probability"] - first_res["probability"]) < 1e-9


# =============================================================================
# PHASE 20: App Factory / Import Safety
# =============================================================================

def test_integration_app_factory_safety():
    """Verifies create_app can be imported and executed without launching servers or modifying state."""
    from member3b.common.api import create_app as factory
    app = factory()
    assert app is not None
    assert app.name == "member3b.common.api"
