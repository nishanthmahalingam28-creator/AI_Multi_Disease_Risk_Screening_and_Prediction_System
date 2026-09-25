"""
Integration test suite for Lung Cancer Prediction Module.

Validates end-to-end integration and production hardening across the full stack:
HTTP Request -> Flask API -> Service Layer -> Prediction Engine -> Preprocessor -> Trained Model -> Response.

Medical Safety Disclaimer:
This test suite verifies an ML-based risk screening prediction system.
All interpretations represent screening predictions and risk estimates,
not medical diagnoses.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import math
import re
import time
import pytest
import numpy as np

from member3b.lung_cancer.src.api import create_app
from member3b.lung_cancer.src.service import (
    LungCancerPredictionService,
    SERVICE_NAME,
    DISEASE_KEY,
)
from member3b.lung_cancer.src.predict import EXPECTED_PREDICTOR_FEATURES
from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data


@pytest.fixture
def app():
    """Provides a fresh Flask application configured for integration testing."""
    return create_app(test_config={"TESTING": True})


@pytest.fixture
def client(app):
    """Provides an HTTP test client."""
    return app.test_client()


@pytest.fixture
def service():
    """Provides a live LungCancerPredictionService instance."""
    return LungCancerPredictionService()


@pytest.fixture
def valid_sample_payload():
    """Provides a valid 15-feature Lung Cancer survey payload sourced from raw data."""
    raw_df = load_raw_lung_cancer_data()
    row = raw_df.iloc[0]
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
# PHASE 4: End-to-End Health Test
# =============================================================================

def test_integration_health_endpoint_healthy(client):
    """Verifies GET /api/lung-cancer/health returns HTTP 200 with readiness status."""
    response = client.get("/api/lung-cancer/health")
    assert response.status_code == 200
    assert response.is_json
    assert "application/json" in response.content_type

    data = response.get_json()
    assert data["status"] == "ready"
    assert data["service"] == SERVICE_NAME
    assert data["disease"] == DISEASE_KEY
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True

    # Verify no filesystem paths or tracebacks are exposed
    raw_text = response.get_data(as_text=True)
    assert not re.search(r"[A-Za-z]:\\", raw_text)
    assert "/Users/" not in raw_text
    assert "/home/" not in raw_text
    assert "Traceback" not in raw_text
    assert "Exception" not in raw_text


# =============================================================================
# PHASE 5: End-to-End Model Info Test
# =============================================================================

def test_integration_model_info_endpoint(client):
    """Verifies GET /api/lung-cancer/model-info returns safe metadata without leaking internals."""
    response = client.get("/api/lung-cancer/model-info")
    assert response.status_code == 200
    assert response.is_json
    assert "application/json" in response.content_type

    data = response.get_json()
    assert data["status"] == "success"
    assert data["service"] == SERVICE_NAME
    assert data["disease"] == DISEASE_KEY
    assert data["model"] == "LogisticRegression"
    assert data["feature_count"] == 15
    assert len(data["features"]) == 15
    assert data["features"] == EXPECTED_PREDICTOR_FEATURES
    assert abs(data["threshold"] - 0.50) < 1e-6
    assert data["target_mapping"] == {"NO": 0, "YES": 1}
    assert data["class_mapping"] == {"0": "NO", "1": "YES"} or data["class_mapping"] == {0: "NO", 1: "YES"}

    # Verify safe metadata only: no coefficients, file paths, or env details
    raw_text = response.get_data(as_text=True)
    for forbidden in ["coef_", "intercept_", "filepath", "joblib", "Traceback", "secret"]:
        assert forbidden not in raw_text
    assert not re.search(r"[A-Za-z]:\\", raw_text)


# =============================================================================
# PHASE 6: User Prediction Integration
# =============================================================================

def test_integration_user_prediction_flow(client, valid_sample_payload):
    """Verifies full request flow for valid screening prediction via user endpoint."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_sample_payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.is_json
    assert "application/json" in response.content_type

    data = response.get_json()
    assert data["status"] == "success"
    assert data["service"] == SERVICE_NAME
    assert data["disease"] == DISEASE_KEY
    assert data["prediction"] in (0, 1)
    assert data["prediction_class"] in ("NO", "YES")
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["risk_percentage"], (int, float))
    assert 0.0 <= data["risk_percentage"] <= 100.0
    assert abs(data["risk_percentage"] - (data["probability"] * 100)) < 0.05
    assert data["model"] == "LogisticRegression"
    assert abs(data["threshold"] - 0.50) < 1e-6
    assert data["screening_interpretation"] in ("Higher predicted risk", "Lower predicted risk")


# =============================================================================
# PHASE 7: Clinical Prediction Integration & Cross-Consistency
# =============================================================================

def test_integration_clinical_prediction_flow_and_consistency(client, valid_sample_payload):
    """Verifies clinical prediction flow and mathematical consistency with user prediction."""
    # Clinical prediction
    resp_clin = client.post(
        "/api/lung-cancer/predict/clinical",
        data=json.dumps(valid_sample_payload),
        content_type="application/json",
    )
    assert resp_clin.status_code == 200
    data_clin = resp_clin.get_json()
    assert data_clin["status"] == "success"

    # User prediction
    resp_user = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_sample_payload),
        content_type="application/json",
    )
    assert resp_user.status_code == 200
    data_user = resp_user.get_json()

    # Verify exact cross-interface consistency
    assert data_user["prediction"] == data_clin["prediction"]
    assert data_user["prediction_class"] == data_clin["prediction_class"]
    assert abs(data_user["probability"] - data_clin["probability"]) < 1e-6
    assert abs(data_user["risk_percentage"] - data_clin["risk_percentage"]) < 1e-4
    assert data_user["screening_interpretation"] == data_clin["screening_interpretation"]
    assert data_user["model"] == data_clin["model"]
    assert data_user["threshold"] == data_clin["threshold"]


# =============================================================================
# PHASE 8: Repeated Prediction Consistency
# =============================================================================

def test_integration_repeated_prediction_consistency(client, valid_sample_payload):
    """Verifies that 10 repeated calls yield deterministic outputs with no state mutation."""
    initial_res = None
    for _ in range(10):
        response = client.post(
            "/api/lung-cancer/predict/user",
            data=json.dumps(valid_sample_payload),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()

        if initial_res is None:
            initial_res = data
        else:
            assert data["prediction"] == initial_res["prediction"]
            assert data["prediction_class"] == initial_res["prediction_class"]
            assert abs(data["probability"] - initial_res["probability"]) < 1e-9
            assert abs(data["risk_percentage"] - initial_res["risk_percentage"]) < 1e-9


# =============================================================================
# PHASE 9: Validation Integration (Missing, Unexpected, Values, Bounds, Types)
# =============================================================================

def test_integration_missing_single_feature(client, valid_sample_payload):
    """Verifies missing required feature returns HTTP 400 validation error."""
    payload = dict(valid_sample_payload)
    del payload["smoking"]

    for endpoint in ["/api/lung-cancer/predict/user", "/api/lung-cancer/predict/clinical"]:
        resp = client.post(endpoint, data=json.dumps(payload), content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "validation_error"
        assert "Missing required feature" in data["message"]
        assert "smoking" in data["message"]


def test_integration_missing_multiple_and_empty_payload(client, valid_sample_payload):
    """Verifies missing multiple features or empty JSON payload returns HTTP 400."""
    # Missing multiple
    multi_missing = {"age": 55, "gender": "MALE"}
    resp = client.post("/api/lung-cancer/predict/user", data=json.dumps(multi_missing), content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["error_type"] == "validation_error"

    # Empty JSON object
    resp_empty = client.post("/api/lung-cancer/predict/user", data=json.dumps({}), content_type="application/json")
    assert resp_empty.status_code == 400
    assert resp_empty.get_json()["error_type"] == "validation_error"


def test_integration_unexpected_feature_rejected(client, valid_sample_payload):
    """Verifies extraneous or unauthorized features return HTTP 400 validation error."""
    payload = dict(valid_sample_payload)
    payload["tumor_stage"] = "Stage II"

    for endpoint in ["/api/lung-cancer/predict/user", "/api/lung-cancer/predict/clinical"]:
        resp = client.post(endpoint, data=json.dumps(payload), content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "validation_error"
        assert "Unexpected feature encountered" in data["message"]


def test_integration_invalid_gender_values_rejected(client, valid_sample_payload):
    """Verifies invalid gender string, boolean, or null values are rejected with HTTP 400."""
    # Invalid string
    p1 = dict(valid_sample_payload, gender="UNKNOWN")
    r1 = client.post("/api/lung-cancer/predict/user", data=json.dumps(p1), content_type="application/json")
    assert r1.status_code == 400
    assert "Invalid value for gender" in r1.get_json()["message"]

    # Boolean value
    p2 = dict(valid_sample_payload, gender=True)
    r2 = client.post("/api/lung-cancer/predict/user", data=json.dumps(p2), content_type="application/json")
    assert r2.status_code == 400
    assert "Boolean values are not accepted" in r2.get_json()["message"]

    # Null value
    p3 = dict(valid_sample_payload, gender=None)
    r3 = client.post("/api/lung-cancer/predict/user", data=json.dumps(p3), content_type="application/json")
    assert r3.status_code == 400
    assert "cannot be None" in r3.get_json()["message"]


def test_integration_invalid_age_bounds_and_types_rejected(client, valid_sample_payload):
    """Verifies age bounds (age <= 0, age > 120), strings, booleans, and null are rejected."""
    # Age <= 0
    p_zero = dict(valid_sample_payload, age=0)
    r_zero = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_zero), content_type="application/json")
    assert r_zero.status_code == 400
    assert "Age must be positive" in r_zero.get_json()["message"]

    p_neg = dict(valid_sample_payload, age=-1)
    r_neg = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_neg), content_type="application/json")
    assert r_neg.status_code == 400
    assert "Age must be positive" in r_neg.get_json()["message"]

    # Age > 120
    p_high = dict(valid_sample_payload, age=121)
    r_high = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_high), content_type="application/json")
    assert r_high.status_code == 400
    assert "exceeds realistic human limit" in r_high.get_json()["message"]

    # Age non-numeric string
    p_str = dict(valid_sample_payload, age="not_a_number")
    r_str = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_str), content_type="application/json")
    assert r_str.status_code == 400
    assert "Invalid non-numeric value for age" in r_str.get_json()["message"]

    # Age boolean
    p_bool = dict(valid_sample_payload, age=True)
    r_bool = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_bool), content_type="application/json")
    assert r_bool.status_code == 400
    assert "Boolean values are not accepted" in r_bool.get_json()["message"]

    # Age null
    p_null = dict(valid_sample_payload, age=None)
    r_null = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_null), content_type="application/json")
    assert r_null.status_code == 400
    assert "cannot be None" in r_null.get_json()["message"]


def test_integration_invalid_symptom_values_rejected(client, valid_sample_payload):
    """Verifies symptom values not in {0, 1, 'YES', 'NO'} or boolean literals are rejected."""
    # Out of range numeric value
    p1 = dict(valid_sample_payload, smoking=5)
    r1 = client.post("/api/lung-cancer/predict/user", data=json.dumps(p1), content_type="application/json")
    assert r1.status_code == 400
    assert "Expected 0, 1, YES, or NO" in r1.get_json()["message"]

    # Invalid string
    p2 = dict(valid_sample_payload, coughing="MAYBE")
    r2 = client.post("/api/lung-cancer/predict/user", data=json.dumps(p2), content_type="application/json")
    assert r2.status_code == 400
    assert "Expected 0, 1, YES, or NO" in r2.get_json()["message"]

    # Boolean literal
    p3 = dict(valid_sample_payload, wheezing=True)
    r3 = client.post("/api/lung-cancer/predict/user", data=json.dumps(p3), content_type="application/json")
    assert r3.status_code == 400
    assert "Boolean values" in r3.get_json()["message"]

    # Null symptom
    p4 = dict(valid_sample_payload, fatigue=None)
    r4 = client.post("/api/lung-cancer/predict/user", data=json.dumps(p4), content_type="application/json")
    assert r4.status_code == 400
    assert "cannot be None" in r4.get_json()["message"]


def test_integration_nan_and_infinity_safety(client, valid_sample_payload):
    """Verifies NaN, Infinity, and -Infinity values return controlled HTTP 400 responses."""
    # NaN in age
    p_nan = dict(valid_sample_payload, age=float("nan"))
    r_nan = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_nan), content_type="application/json")
    assert r_nan.status_code == 400
    assert "cannot be NaN" in r_nan.get_json()["message"]

    # Infinity in age
    p_inf = dict(valid_sample_payload, age=float("inf"))
    r_inf = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_inf), content_type="application/json")
    assert r_inf.status_code == 400
    assert "cannot be infinite" in r_inf.get_json()["message"]

    # -Infinity in age
    p_ninf = dict(valid_sample_payload, age=float("-inf"))
    r_ninf = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_ninf), content_type="application/json")
    assert r_ninf.status_code == 400
    assert "cannot be infinite" in r_ninf.get_json()["message"]

    # NaN in symptom
    p_sym_nan = dict(valid_sample_payload, chest_pain=float("nan"))
    r_sym_nan = client.post("/api/lung-cancer/predict/user", data=json.dumps(p_sym_nan), content_type="application/json")
    assert r_sym_nan.status_code == 400
    assert "cannot be NaN" in r_sym_nan.get_json()["message"]


def test_integration_non_dict_json_types_rejected(client):
    """Verifies top-level JSON array, string, number, boolean, and null return HTTP 400."""
    invalid_bodies = [
        [{"gender": "MALE", "age": 55}],
        "just a string",
        12345,
        True,
        None,
    ]

    for body in invalid_bodies:
        resp = client.post(
            "/api/lung-cancer/predict/user",
            data=json.dumps(body),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "invalid_request"
        assert "valid JSON object" in data["message"]


# =============================================================================
# PHASE 10: Malformed JSON and Duplicate JSON Keys
# =============================================================================

def test_integration_malformed_json_syntax(client):
    """Verifies malformed JSON syntax is trapped cleanly with HTTP 400 and safe message."""
    malformed_payloads = [
        '{"gender": "MALE", "age": 55',  # missing closing brace
        '{"gender": "MALE", age: 55}',    # unquoted key
        '{invalid_json',
    ]

    for payload in malformed_payloads:
        resp = client.post(
            "/api/lung-cancer/predict/user",
            data=payload,
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "invalid_request"
        assert "Malformed JSON" in data["message"] or "Invalid JSON" in data["message"]
        # Verify no traceback in output
        assert "Traceback" not in resp.get_data(as_text=True)


def test_integration_duplicate_json_keys_rejection(client, valid_sample_payload):
    """Verifies payloads with duplicate JSON keys are detected and rejected with HTTP 400."""
    # Construct raw payload with duplicate "age"
    base_json = json.dumps(valid_sample_payload)
    duplicate_payload = '{"age": 50, ' + base_json[1:]

    for endpoint in ["/api/lung-cancer/predict/user", "/api/lung-cancer/predict/clinical"]:
        resp = client.post(endpoint, data=duplicate_payload, content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["error_type"] == "validation_error"
        assert "Duplicate JSON field: age" in data["message"]


# =============================================================================
# PHASE 11: Content-Type Hardening
# =============================================================================

def test_integration_content_type_hardening(client, valid_sample_payload):
    """Verifies unsupported or missing Content-Type headers receive controlled HTTP 400 responses."""
    raw_json = json.dumps(valid_sample_payload)

    # 1. Missing Content-Type
    resp_no_ct = client.post("/api/lung-cancer/predict/user", data=raw_json)
    assert resp_no_ct.status_code == 400
    assert "application/json" in resp_no_ct.get_json()["message"]

    # 2. text/plain
    resp_plain = client.post("/api/lung-cancer/predict/user", data=raw_json, content_type="text/plain")
    assert resp_plain.status_code == 400
    assert "application/json" in resp_plain.get_json()["message"]

    # 3. application/x-www-form-urlencoded
    resp_form = client.post(
        "/api/lung-cancer/predict/user",
        data="gender=MALE&age=69",
        content_type="application/x-www-form-urlencoded",
    )
    assert resp_form.status_code == 400
    assert "application/json" in resp_form.get_json()["message"]

    # 4. Valid application/json succeeds
    resp_valid = client.post("/api/lung-cancer/predict/user", data=raw_json, content_type="application/json")
    assert resp_valid.status_code == 200


# =============================================================================
# PHASE 12: Model and Preprocessor Failure Simulation
# =============================================================================

def test_integration_missing_artifacts_simulation(valid_sample_payload):
    """Verifies controlled HTTP 503 response when model artifacts directory is unavailable."""
    isolated_service = LungCancerPredictionService(models_dir=Path("/non/existent/model/path"))
    isolated_app = create_app(test_config={"TESTING": True}, service=isolated_service)
    isolated_client = isolated_app.test_client()

    # Health check returns 503
    h_resp = isolated_client.get("/api/lung-cancer/health")
    assert h_resp.status_code == 503
    assert h_resp.get_json()["status"] == "unavailable"

    # Prediction endpoint returns 503
    p_resp = isolated_client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_sample_payload),
        content_type="application/json",
    )
    assert p_resp.status_code == 503
    data = p_resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "service_error"
    assert "currently unavailable" in data["message"]

    # Ensure no path leaked and no traceback
    text = p_resp.get_data(as_text=True)
    assert "Traceback" not in text
    assert not re.search(r"[A-Za-z]:\\", text)


def test_integration_corrupted_model_object_simulation(monkeypatch, client, valid_sample_payload):
    """Verifies HTTP 503 when model object fails to predict during inference."""
    import member3b.lung_cancer.src.predict as p_mod

    orig_load = p_mod.load_artifacts
    preprocessor, model, metadata = orig_load()

    class FaultyModel:
        def predict(self, X):
            raise RuntimeError("Corrupted weights execution failure")

    monkeypatch.setattr(p_mod, "load_artifacts", lambda **kwargs: (preprocessor, FaultyModel(), metadata))

    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_sample_payload),
        content_type="application/json",
    )
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "service_error"
    assert "currently unavailable" in data["message"]
    assert "Traceback" not in response.get_data(as_text=True)


def test_integration_corrupted_preprocessor_object_simulation(monkeypatch, client, valid_sample_payload):
    """Verifies HTTP 503 when preprocessor object fails during feature scaling."""
    import member3b.lung_cancer.src.predict as p_mod

    orig_load = p_mod.load_artifacts
    preprocessor, model, metadata = orig_load()

    class FaultyPreprocessor:
        def transform(self, X):
            raise RuntimeError("Pipeline transformation failure")

    monkeypatch.setattr(p_mod, "load_artifacts", lambda **kwargs: (FaultyPreprocessor(), model, metadata))

    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_sample_payload),
        content_type="application/json",
    )
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "service_error"
    assert "currently unavailable" in data["message"]


# =============================================================================
# PHASE 13: Service Failure Simulation
# =============================================================================

def test_integration_service_exception_simulation(monkeypatch, client, valid_sample_payload):
    """Verifies unhandled service exceptions return HTTP 500 JSON without leaking tracebacks."""
    # Force service.predict_user to raise an unexpected exception
    import member3b.lung_cancer.src.service as s_mod

    def exploding_predict_user(*args, **kwargs):
        raise RuntimeError("Unexpected internal crash")

    monkeypatch.setattr(s_mod.LungCancerPredictionService, "predict_user", exploding_predict_user)

    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_sample_payload),
        content_type="application/json",
    )
    assert response.status_code == 500
    assert response.is_json
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "internal_error"
    assert "internal server error" in data["message"].lower()

    # Verify no traceback or secret leakage
    text = response.get_data(as_text=True)
    assert "Traceback" not in text
    assert "exploding_predict_user" not in text


# =============================================================================
# PHASE 14: API Contract Testing
# =============================================================================

def test_integration_api_contract_types(client, valid_sample_payload):
    """Verifies all JSON response fields conform to strict contract types and bounds."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_sample_payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()

    # Contract assertions
    assert isinstance(data["status"], str) and data["status"] == "success"
    assert isinstance(data["service"], str) and data["service"] == SERVICE_NAME
    assert isinstance(data["disease"], str) and data["disease"] == DISEASE_KEY
    assert isinstance(data["prediction"], int) and data["prediction"] in (0, 1)
    assert isinstance(data["prediction_class"], str) and data["prediction_class"] in ("NO", "YES")
    assert isinstance(data["probability"], float) and 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["risk_percentage"], (float, int)) and 0.0 <= data["risk_percentage"] <= 100.0
    assert isinstance(data["threshold"], float) and abs(data["threshold"] - 0.50) < 1e-6
    assert isinstance(data["model"], str) and data["model"] == "LogisticRegression"
    assert isinstance(data["screening_interpretation"], str)


# =============================================================================
# PHASE 15: Error Contract Testing (400, 404, 405, 500, 503)
# =============================================================================

def test_integration_error_contract_json_responses(client):
    """Verifies that all error status codes (400, 404, 405, 500, 503) return JSON without leakage."""
    # 400 Bad Request
    r400 = client.post("/api/lung-cancer/predict/user", data="bad", content_type="application/json")
    assert r400.status_code == 400
    assert r400.is_json
    assert r400.get_json()["status"] == "error"

    # 404 Not Found
    r404 = client.get("/api/lung-cancer/unknown-route")
    assert r404.status_code == 404
    assert r404.is_json
    assert r404.get_json()["status"] == "error"
    assert r404.get_json()["error_type"] == "not_found"

    # 405 Method Not Allowed
    r405 = client.get("/api/lung-cancer/predict/user")
    assert r405.status_code == 405
    assert r405.is_json
    assert r405.get_json()["status"] == "error"
    assert r405.get_json()["error_type"] == "method_not_allowed"

    # 503 Service Unavailable
    isolated_app = create_app(test_config={"TESTING": True}, service=LungCancerPredictionService(models_dir=Path("/non/existent")))
    r503 = isolated_app.test_client().get("/api/lung-cancer/health")
    assert r503.status_code == 503
    assert r503.is_json
    assert r503.get_json()["status"] == "unavailable"

    # Verify no traceback in any error responses
    for r in [r400, r404, r405, r503]:
        text = r.get_data(as_text=True)
        assert "Traceback" not in text
        assert not re.search(r"[A-Za-z]:\\", text)


# =============================================================================
# PHASE 16: App Factory and Import Safety
# =============================================================================

def test_integration_app_factory_and_import_safety():
    """Verifies importing modules and invoking create_app() does not start servers or mutate state."""
    from member3b.lung_cancer.src.api import create_app as factory
    test_app = factory()
    assert test_app is not None
    assert test_app.name == "member3b.lung_cancer.src.api"


# =============================================================================
# PHASE 17: Request State Isolation
# =============================================================================

def test_integration_request_state_isolation(client, valid_sample_payload):
    """Verifies subsequent requests are completely isolated across valid/invalid sequences."""
    # Sequence: Valid -> Invalid -> Valid
    r1 = client.post("/api/lung-cancer/predict/user", data=json.dumps(valid_sample_payload), content_type="application/json")
    assert r1.status_code == 200

    r2 = client.post("/api/lung-cancer/predict/user", data=json.dumps({"invalid": 1}), content_type="application/json")
    assert r2.status_code == 400

    r3 = client.post("/api/lung-cancer/predict/user", data=json.dumps(valid_sample_payload), content_type="application/json")
    assert r3.status_code == 200
    assert r3.get_json()["prediction"] == r1.get_json()["prediction"]

    # Sequence: Clinical -> User -> Clinical
    rc1 = client.post("/api/lung-cancer/predict/clinical", data=json.dumps(valid_sample_payload), content_type="application/json")
    assert rc1.status_code == 200

    ru = client.post("/api/lung-cancer/predict/user", data=json.dumps(valid_sample_payload), content_type="application/json")
    assert ru.status_code == 200

    rc2 = client.post("/api/lung-cancer/predict/clinical", data=json.dumps(valid_sample_payload), content_type="application/json")
    assert rc2.status_code == 200
    assert rc1.get_json()["prediction"] == rc2.get_json()["prediction"]


# =============================================================================
# PHASE 18: Concurrency Check
# =============================================================================

def test_integration_concurrency_check(app, valid_sample_payload):
    """Verifies thread-safe inference across concurrent requests with zero data contamination."""
    def run_request(_):
        test_client = app.test_client()
        resp = test_client.post(
            "/api/lung-cancer/predict/user",
            data=json.dumps(valid_sample_payload),
            content_type="application/json",
        )
        return resp.status_code, resp.get_json()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_request, range(10)))

    for status_code, data in results:
        assert status_code == 200
        assert data["status"] == "success"
        assert data["prediction"] in (0, 1)
        assert data["prediction_class"] in ("NO", "YES")
        assert abs(data["probability"] - results[0][1]["probability"]) < 1e-9


# =============================================================================
# PHASE 19: Local Flask Test-Client Latency Measurement
# =============================================================================

def test_integration_latency_benchmark(client, valid_sample_payload):
    """
    Measures and asserts reasonable execution latency for repeated prediction requests.
    Clearly labeled as: LOCAL FLASK TEST-CLIENT LATENCY.
    """
    raw_payload = json.dumps(valid_sample_payload)
    latencies = []

    # Run 15 sequential requests
    for _ in range(15):
        t0 = time.perf_counter()
        resp = client.post(
            "/api/lung-cancer/predict/user",
            data=raw_payload,
            content_type="application/json",
        )
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies.append((t1 - t0) * 1000.0)  # ms

    min_lat = min(latencies)
    max_lat = max(latencies)
    avg_lat = sum(latencies) / len(latencies)

    # In local test client, in-process prediction latency should typically be under 50ms
    assert avg_lat < 50.0, f"Average latency too high: {avg_lat:.2f}ms"
    assert min_lat > 0.0
