"""
Step 18 — Member 3B Production Deployment Readiness & WSGI Configuration Test Suite.

Verifies:
- Production WSGI entry point importability and PEP 3333 compliance
- Production configuration, safe defaults (DEBUG=False), and environment overrides
- Host (0.0.0.0) and Port validation (valid, invalid, boundary cases)
- Model artifact availability on disk and working-directory independence
- Lightweight health check endpoint (HTTP 200) under ready state
- Readiness failure simulation (HTTP 503) when any model or preprocessor is unavailable
- Security verification: controlled JSON error responses (400, 404, 405, 500, 503) without stack traces
- Zero environment variable, credentials, or filesystem path leakage
- Server restart idempotence and model artifact / raw dataset immutability
- Full regression for both existing disease APIs and unified Member 3B endpoints
"""

from typing import Dict, Any
from pathlib import Path
import hashlib
import io
import json
import os
import pytest
from wsgiref.validate import validator

from member3b.common.config import get_config, validate_port, AppConfig
from member3b.common.api import create_app
from member3b.common.service import CancerPredictionService
import member3b.app as wsgi_module
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data
from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data
from member3b.breast_cancer.src.service import BreastCancerPredictionService
from member3b.lung_cancer.src.service import LungCancerPredictionService


@pytest.fixture
def app_client():
    """Provides a test client configured for testing."""
    test_app = create_app(test_config={"TESTING": True})
    return test_app.test_client()


@pytest.fixture
def valid_breast_payload() -> Dict[str, Any]:
    """Provides a valid Breast Cancer feature payload from dataset row 0."""
    df = load_raw_breast_cancer_data()
    feature_cols = [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
    return {col: float(df.iloc[0][col]) for col in feature_cols}


@pytest.fixture
def valid_lung_payload() -> Dict[str, Any]:
    """Provides a valid Lung Cancer feature payload from dataset row 0."""
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
# 1. WSGI Entry Point & PEP 3333 Protocol Compliance
# =============================================================================

def test_wsgi_entrypoint_import():
    """Verifies that member3b.app exposes an importable WSGI application object."""
    assert hasattr(wsgi_module, "app"), "member3b.app must expose an 'app' attribute"
    assert hasattr(wsgi_module, "config"), "member3b.app must expose a 'config' attribute"
    assert callable(wsgi_module.app), "WSGI application object must be callable"
    assert hasattr(wsgi_module.app, "wsgi_app"), "Flask application must provide wsgi_app callable"
    assert wsgi_module.app.debug is False, "WSGI application must not have debug enabled by default"


def test_wsgi_pep3333_compliance():
    """Verifies standard PEP 3333 WSGI compliance using wsgiref validator."""
    app = wsgi_module.app
    validated_app = validator(app.wsgi_app)

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

    status_capture = []
    headers_capture = []

    def start_response(status, response_headers, exc_info=None):
        status_capture.append(status)
        headers_capture.append(response_headers)

    response_iter = validated_app(environ, start_response)
    try:
        body = b"".join(response_iter)
    finally:
        if hasattr(response_iter, "close"):
            response_iter.close()

    assert len(status_capture) == 1
    assert "200 OK" in status_capture[0]
    payload = json.loads(body.decode("utf-8"))
    assert payload.get("status") == "ready"


# =============================================================================
# 2. Production Configuration & Environment Variable Validation
# =============================================================================

def test_production_config_defaults():
    """Verifies safe production defaults: localhost, port 5000, debug=False."""
    cfg = get_config()
    assert cfg.HOST == "127.0.0.1"
    assert cfg.PORT == 5000
    assert cfg.DEBUG is False
    assert cfg.ENV == "production"
    assert cfg.TESTING is False


def test_production_config_env_overrides(monkeypatch):
    """Verifies production platform environment variables properly override defaults."""
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "8000")
    monkeypatch.setenv("FLASK_DEBUG", "false")
    monkeypatch.setenv("FLASK_ENV", "production")

    cfg = get_config()
    assert cfg.HOST == "0.0.0.0"
    assert cfg.PORT == 8000
    assert cfg.DEBUG is False
    assert cfg.ENV == "production"


def test_validate_port_valid():
    """Verifies port validator accepts valid TCP integer and string ports."""
    assert validate_port(5000) == 5000
    assert validate_port("8080") == 8080
    assert validate_port(1) == 1
    assert validate_port(65535) == 65535
    assert validate_port("65535") == 65535


def test_validate_port_invalid():
    """Verifies port validator rejects negative, zero, out-of-range, and non-integer values."""
    for invalid_val in [-1, 0, 65536, 99999, "invalid", "", None]:
        with pytest.raises(ValueError):
            validate_port(invalid_val)


def test_get_config_strict_port_validation(monkeypatch):
    """Verifies get_config with strict=True raises ValueError on invalid port values."""
    monkeypatch.setenv("PORT", "-1")
    with pytest.raises(ValueError):
        get_config(strict=True)

    monkeypatch.setenv("PORT", "99999")
    with pytest.raises(ValueError):
        get_config(strict=True)

    monkeypatch.setenv("PORT", "not_a_port")
    with pytest.raises(ValueError):
        get_config(strict=True)


def test_get_config_safe_fallback_on_invalid_port(monkeypatch):
    """Verifies get_config with strict=False safely falls back to default 5000."""
    monkeypatch.setenv("PORT", "-1")
    cfg = get_config(strict=False)
    assert cfg.PORT == 5000

    monkeypatch.setenv("PORT", "99999")
    cfg = get_config(strict=False)
    assert cfg.PORT == 5000

    monkeypatch.setenv("PORT", "invalid")
    cfg = get_config(strict=False)
    assert cfg.PORT == 5000


# =============================================================================
# 3. Model Artifact Availability & Working Directory Independence
# =============================================================================

def test_model_artifacts_available():
    """Verifies all four required model and preprocessor artifacts exist and are non-empty."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    expected_artifacts = [
        repo_root / "member3b" / "breast_cancer" / "models" / "breast_cancer_model.joblib",
        repo_root / "member3b" / "breast_cancer" / "models" / "breast_cancer_preprocessor.joblib",
        repo_root / "member3b" / "lung_cancer" / "models" / "lung_cancer_model.joblib",
        repo_root / "member3b" / "lung_cancer" / "models" / "lung_cancer_preprocessor.joblib",
    ]
    for path in expected_artifacts:
        assert path.exists(), f"Artifact missing: {path}"
        assert path.is_file(), f"Artifact path is not a file: {path}"
        assert path.stat().st_size > 0, f"Artifact file is empty: {path}"


def test_working_directory_independence(monkeypatch, tmp_path, valid_breast_payload, valid_lung_payload):
    """Verifies services and predictions function when current working directory is external."""
    monkeypatch.chdir(tmp_path)
    # Instantiate service from external directory
    svc = CancerPredictionService()
    health = svc.health_check()
    assert health.get("status") == "ready"

    # Verify Breast prediction works from external CWD
    breast_res = svc.predict("breast_cancer", valid_breast_payload)
    assert breast_res.get("status") == "success"

    # Verify Lung prediction works from external CWD
    lung_res = svc.predict("lung_cancer", valid_lung_payload)
    assert lung_res.get("status") == "success"


# =============================================================================
# 4. Health Check & Readiness Failure Simulation
# =============================================================================

def test_health_check_ready_state(app_client):
    """Verifies GET /api/member3b/health returns 200 OK when services are ready."""
    resp = app_client.get("/api/member3b/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ready"
    assert data["service"] == "member3b-cancer-prediction"
    assert data["diseases"]["breast_cancer"]["model_loaded"] is True
    assert data["diseases"]["breast_cancer"]["preprocessor_loaded"] is True
    assert data["diseases"]["lung_cancer"]["model_loaded"] is True
    assert data["diseases"]["lung_cancer"]["preprocessor_loaded"] is True


def test_readiness_failure_breast_model_unavailable(monkeypatch):
    """Verifies GET /api/member3b/health returns 503 when Breast model is unavailable."""
    def mock_breast_health():
        return {
            "status": "unhealthy",
            "disease": "breast_cancer",
            "model_loaded": False,
            "preprocessor_loaded": True,
        }
    monkeypatch.setattr(BreastCancerPredictionService, "health_check", lambda self: mock_breast_health())

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()
    resp = client.get("/api/member3b/health")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["diseases"]["breast_cancer"]["model_loaded"] is False


def test_readiness_failure_breast_preprocessor_unavailable(monkeypatch):
    """Verifies GET /api/member3b/health returns 503 when Breast preprocessor is unavailable."""
    def mock_breast_health():
        return {
            "status": "unhealthy",
            "disease": "breast_cancer",
            "model_loaded": True,
            "preprocessor_loaded": False,
        }
    monkeypatch.setattr(BreastCancerPredictionService, "health_check", lambda self: mock_breast_health())

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()
    resp = client.get("/api/member3b/health")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["diseases"]["breast_cancer"]["preprocessor_loaded"] is False


def test_readiness_failure_lung_model_unavailable(monkeypatch):
    """Verifies GET /api/member3b/health returns 503 when Lung model is unavailable."""
    def mock_lung_health():
        return {
            "status": "unavailable",
            "service": "lung-cancer-prediction",
            "disease": "lung_cancer",
            "model_loaded": False,
            "preprocessor_loaded": True,
        }
    monkeypatch.setattr(LungCancerPredictionService, "health_check", lambda self: mock_lung_health())

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()
    resp = client.get("/api/member3b/health")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["diseases"]["lung_cancer"]["model_loaded"] is False


def test_readiness_failure_lung_preprocessor_unavailable(monkeypatch):
    """Verifies GET /api/member3b/health returns 503 when Lung preprocessor is unavailable."""
    def mock_lung_health():
        return {
            "status": "unavailable",
            "service": "lung-cancer-prediction",
            "disease": "lung_cancer",
            "model_loaded": True,
            "preprocessor_loaded": False,
        }
    monkeypatch.setattr(LungCancerPredictionService, "health_check", lambda self: mock_lung_health())

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()
    resp = client.get("/api/member3b/health")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["diseases"]["lung_cancer"]["preprocessor_loaded"] is False


# =============================================================================
# 5. Security & Controlled Error Responses (400, 404, 405, 500, 503)
# =============================================================================

def test_security_400_bad_request(app_client):
    """Verifies 400 response is valid JSON with no stack trace or path leakage."""
    resp = app_client.post(
        "/api/member3b/predict",
        data="invalid non-json text",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert "Traceback" not in resp.get_data(as_text=True)


def test_security_404_not_found(app_client):
    """Verifies 404 response is valid JSON with no server implementation details."""
    resp = app_client.get("/api/member3b/nonexistent_endpoint")
    assert resp.status_code == 404
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "not_found"
    assert "Traceback" not in resp.get_data(as_text=True)


def test_security_405_method_not_allowed(app_client):
    """Verifies 405 response is valid JSON."""
    resp = app_client.post("/api/member3b/health", json={})
    assert resp.status_code == 405
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "method_not_allowed"


def test_security_500_internal_error_handling():
    """Verifies 500 internal server error returns clean JSON without traceback."""
    test_app = create_app(test_config={"TESTING": False, "DEBUG": False})

    @test_app.route("/api/test-error-500", methods=["GET"])
    def error_route():
        raise RuntimeError("Simulated unhandled internal fault")

    client = test_app.test_client()
    resp = client.get("/api/test-error-500")
    assert resp.status_code == 500
    assert resp.is_json
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "internal_error"
    assert "RuntimeError" not in resp.get_data(as_text=True)
    assert "Traceback" not in resp.get_data(as_text=True)


def test_security_zero_env_var_leakage(app_client, monkeypatch, valid_breast_payload):
    """Verifies environment variables and secrets are never reflected in API responses."""
    secret_token = "SECRET_TOKEN_XYZ_9876543210"
    monkeypatch.setenv("SECRET_KEY", secret_token)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost:5432/db")

    endpoints = [
        ("/api/member3b/health", "GET", None),
        ("/api/member3b/model-info", "GET", None),
        ("/api/member3b/predict", "POST", {"disease": "breast_cancer", "data": valid_breast_payload}),
        ("/api/breast-cancer/health", "GET", None),
        ("/api/lung-cancer/health", "GET", None),
    ]

    for path, method, payload in endpoints:
        if method == "GET":
            resp = app_client.get(path)
        else:
            resp = app_client.post(path, json=payload)
        text = resp.get_data(as_text=True)
        assert secret_token not in text, f"Secret leaked in response from {path}"
        assert "postgres://" not in text, f"Database URL leaked in response from {path}"


# =============================================================================
# 6. Server Restart Idempotence & Artifact Immutability
# =============================================================================

def test_restart_idempotence_and_artifact_immutability(valid_breast_payload, valid_lung_payload):
    """Verifies repeated server initialization and inference leaves artifacts unchanged."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tracked_files = [
        repo_root / "dataset" / "breast.csv",
        repo_root / "dataset" / "survey_lung_cancer.csv",
        repo_root / "member3b" / "breast_cancer" / "models" / "breast_cancer_model.joblib",
        repo_root / "member3b" / "breast_cancer" / "models" / "breast_cancer_preprocessor.joblib",
        repo_root / "member3b" / "lung_cancer" / "models" / "lung_cancer_model.joblib",
        repo_root / "member3b" / "lung_cancer" / "models" / "lung_cancer_preprocessor.joblib",
    ]

    # Compute baseline hashes
    initial_hashes = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in tracked_files}

    # Simulate 3 server restart cycles
    for cycle in range(3):
        cycle_app = create_app(test_config={"TESTING": True})
        cycle_client = cycle_app.test_client()

        h_resp = cycle_client.get("/api/member3b/health")
        assert h_resp.status_code == 200

        b_resp = cycle_client.post(
            "/api/member3b/predict",
            json={"disease": "breast_cancer", "data": valid_breast_payload},
        )
        assert b_resp.status_code == 200

        l_resp = cycle_client.post(
            "/api/member3b/predict",
            json={"disease": "lung_cancer", "data": valid_lung_payload},
        )
        assert l_resp.status_code == 200

    # Verify hashes are completely unchanged after all cycles
    for f in tracked_files:
        current_hash = hashlib.sha256(f.read_bytes()).hexdigest()
        assert current_hash == initial_hashes[f], f"File modified during execution: {f}"


# =============================================================================
# 7. Existing API Regression
# =============================================================================

def test_existing_apis_regression_breast(app_client, valid_breast_payload):
    """Verifies all 4 existing Breast Cancer endpoints continue to function."""
    # Health
    h = app_client.get("/api/breast-cancer/health")
    assert h.status_code == 200
    assert h.get_json()["status"] == "ready"

    # Model Info
    m = app_client.get("/api/breast-cancer/model-info")
    assert m.status_code == 200
    assert m.get_json()["disease"] == "breast_cancer"
    assert "model" in m.get_json()


    # Predict User
    pu = app_client.post("/api/breast-cancer/predict/user", json=valid_breast_payload)
    assert pu.status_code == 200
    assert pu.get_json()["status"] == "success"

    # Predict Clinical
    pc = app_client.post("/api/breast-cancer/predict/clinical", json=valid_breast_payload)
    assert pc.status_code == 200
    assert pc.get_json()["status"] == "success"


def test_existing_apis_regression_lung(app_client, valid_lung_payload):
    """Verifies all 4 existing Lung Cancer endpoints continue to function."""
    # Health
    h = app_client.get("/api/lung-cancer/health")
    assert h.status_code == 200
    assert h.get_json()["status"] == "ready"

    # Model Info
    m = app_client.get("/api/lung-cancer/model-info")
    assert m.status_code == 200
    assert m.get_json()["status"] == "success"

    # Predict User
    pu = app_client.post("/api/lung-cancer/predict/user", json=valid_lung_payload)
    assert pu.status_code == 200
    assert pu.get_json()["status"] == "success"

    # Predict Clinical
    pc = app_client.post("/api/lung-cancer/predict/clinical", json=valid_lung_payload)
    assert pc.status_code == 200
    assert pc.get_json()["status"] == "success"


def test_unified_apis_regression(app_client, valid_breast_payload, valid_lung_payload):
    """Verifies all 3 unified Member 3B endpoints function correctly."""
    # Health
    h = app_client.get("/api/member3b/health")
    assert h.status_code == 200
    assert h.get_json()["status"] == "ready"

    # Model Info
    m = app_client.get("/api/member3b/model-info")
    assert m.status_code == 200
    assert m.get_json()["status"] == "success"

    # Predict Breast
    pb = app_client.post(
        "/api/member3b/predict",
        json={"disease": "breast_cancer", "data": valid_breast_payload},
    )
    assert pb.status_code == 200
    assert pb.get_json()["status"] == "success"
    assert pb.get_json()["disease"] == "breast_cancer"

    # Predict Lung
    pl = app_client.post(
        "/api/member3b/predict",
        json={"disease": "lung_cancer", "data": valid_lung_payload},
    )
    assert pl.status_code == 200
    assert pl.get_json()["status"] == "success"
    assert pl.get_json()["disease"] == "lung_cancer"
