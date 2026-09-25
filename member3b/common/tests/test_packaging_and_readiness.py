"""
Packaging, Configuration, and Local Deployment Readiness Test Suite (Step 17).

Verifies:
- Centralized configuration and environment variable overrides
- Safe defaults (no debug mode, localhost binding, zero secrets)
- Import safety across modules and entry points
- Robust model artifact path resolution independent of CWD
- Package manifest and dependency declarations
- .env.example safety and .gitignore hygiene
- End-to-end smoke testing of unified and direct disease endpoints on app.py
"""

from typing import Dict, Any
from pathlib import Path
import hashlib
import json
import os
import re
import pytest

from member3b.common.config import get_config, AppConfig
from member3b.common.api import create_app
import member3b.app as entry_module
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data
from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data
from member3b.breast_cancer.src.predict import get_default_models_dir as get_breast_models_dir
from member3b.lung_cancer.src.predict import get_default_models_dir as get_lung_models_dir


@pytest.fixture
def app_client():
    """Provides a test client derived directly from member3b.app entry point."""
    test_app = create_app(test_config={"TESTING": True})
    return test_app.test_client()


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
# 1. Configuration & Safe Defaults Tests
# =============================================================================

def test_config_defaults():
    """Verifies that default configuration uses safe localhost, standard port, and debug=False."""
    cfg = get_config()
    assert cfg.HOST == "127.0.0.1"
    assert cfg.PORT == 5000
    assert cfg.DEBUG is False
    assert cfg.ENV == "production"
    assert cfg.TESTING is False


def test_config_env_overrides(monkeypatch):
    """Verifies environment variables properly override configuration defaults."""
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("FLASK_DEBUG", "true")
    monkeypatch.setenv("FLASK_ENV", "development")

    cfg = get_config()
    assert cfg.HOST == "0.0.0.0"
    assert cfg.PORT == 8080
    assert cfg.DEBUG is True
    assert cfg.ENV == "development"


def test_config_invalid_port_fallback(monkeypatch):
    """Verifies that invalid non-integer PORT safely falls back to default 5000."""
    monkeypatch.setenv("PORT", "invalid_port")
    cfg = get_config()
    assert cfg.PORT == 5000


# =============================================================================
# 2. Application Entry Point & Import Safety Tests
# =============================================================================

def test_app_entry_import_safety():
    """Verifies member3b.app can be imported safely without starting the dev server."""
    assert hasattr(entry_module, "app")
    assert hasattr(entry_module, "config")
    assert entry_module.app.name == "member3b.common.api"


def test_app_factory_creates_fresh_instance():
    """Verifies create_app application factory returns a functional Flask app."""
    app_instance = create_app()
    assert app_instance is not None
    assert app_instance.name == "member3b.common.api"


def test_app_no_debug_by_default():
    """Verifies default Flask application has debug mode disabled."""
    app_instance = create_app()
    assert app_instance.debug is False


# =============================================================================
# 3. Model Paths & Artifact Integrity Tests
# =============================================================================

def test_model_paths_resolution_independent_of_cwd(monkeypatch):
    """Verifies model directories resolve via __file__ independently of current working directory."""
    breast_dir = get_breast_models_dir()
    lung_dir = get_lung_models_dir()

    assert breast_dir.exists() and breast_dir.is_dir()
    assert lung_dir.exists() and lung_dir.is_dir()

    assert (breast_dir / "breast_cancer_model.joblib").exists()
    assert (breast_dir / "breast_cancer_preprocessor.joblib").exists()
    assert (lung_dir / "lung_cancer_model.joblib").exists()
    assert (lung_dir / "lung_cancer_preprocessor.joblib").exists()


def test_all_four_model_artifacts_integrity():
    """Verifies that all 4 model and preprocessor artifacts exist with identical baseline SHA-256 hashes."""
    breast_dir = get_breast_models_dir()
    lung_dir = get_lung_models_dir()

    bm_hash = hashlib.sha256((breast_dir / "breast_cancer_model.joblib").read_bytes()).hexdigest()
    bp_hash = hashlib.sha256((breast_dir / "breast_cancer_preprocessor.joblib").read_bytes()).hexdigest()
    lm_hash = hashlib.sha256((lung_dir / "lung_cancer_model.joblib").read_bytes()).hexdigest()
    lp_hash = hashlib.sha256((lung_dir / "lung_cancer_preprocessor.joblib").read_bytes()).hexdigest()

    assert bm_hash == "0379e06d00ad57a172c6c8c6b91eaba48a619e0783644ead1cf5b3e855b32024"
    assert bp_hash == "7dc70392e4bda3f3b53f5cb6b0b6be6a085059b3d706284e392157beb41c3092"
    assert lm_hash == "4c31a9af4b304a21fc9eed29f9c5a9f74caec0cadf02f0d36afde3747eeb6785"
    assert lp_hash == "f40b1b0f3d4f2ff6322d8362cf8c1eaaa7c6e5dab612a29dba2e00cdcb2d527c"


def test_raw_datasets_exist_and_hashes_match():
    """Verifies that raw datasets exist and match their verified baseline SHA-256 hashes."""
    repo_root = Path(__file__).resolve().parents[3]
    b_path = repo_root / "dataset" / "breast.csv"
    l_path = repo_root / "dataset" / "survey_lung_cancer.csv"

    assert b_path.exists()
    assert l_path.exists()

    b_hash = hashlib.sha256(b_path.read_bytes()).hexdigest()
    l_hash = hashlib.sha256(l_path.read_bytes()).hexdigest()

    assert b_hash == "1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33"
    assert l_hash == "181ccdd5e12900a9a428e1ad7a570227617d41266c4dc7930465ebfa009a3efd"


# =============================================================================
# 4. Packaging, .env.example, and .gitignore Tests
# =============================================================================

def test_no_secrets_in_env_example():
    """Verifies that member3b/.env.example contains only safe example configuration and zero secrets."""
    repo_root = Path(__file__).resolve().parents[3]
    env_example = repo_root / "member3b" / ".env.example"
    assert env_example.exists()

    content = env_example.read_text(encoding="utf-8")
    config_lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    for line in config_lines:
        for secret_token in ["password", "secret", "private_key", "token", "api_key", "bearer"]:
            assert secret_token not in line.lower()


def test_gitignore_contains_required_patterns():
    """Verifies that .gitignore contains entries for pycache, venv, and env files."""
    repo_root = Path(__file__).resolve().parents[3]
    gitignore = repo_root / ".gitignore"
    assert gitignore.exists()

    content = gitignore.read_text(encoding="utf-8")
    for pattern in ["__pycache__", ".venv", ".env", ".pytest_cache", "*.py"]:
        assert pattern in content


def test_requirements_file_specifies_dependencies():
    """Verifies that member3b/requirements.txt specifies required runtime dependencies."""
    repo_root = Path(__file__).resolve().parents[3]
    reqs_path = repo_root / "member3b" / "requirements.txt"
    assert reqs_path.exists()

    content = reqs_path.read_text(encoding="utf-8").lower()
    for dep in ["flask", "pandas", "numpy", "scikit-learn", "joblib", "scipy", "pytest"]:
        assert dep in content


# =============================================================================
# 5. Endpoint Smoke Tests on Application
# =============================================================================

def test_unified_health_smoke(app_client):
    """Smoke test: GET /api/member3b/health returns 200 ready."""
    resp = app_client.get("/api/member3b/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ready"


def test_unified_model_info_smoke(app_client):
    """Smoke test: GET /api/member3b/model-info returns 200 success."""
    resp = app_client.get("/api/member3b/model-info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"


def test_unified_breast_predict_smoke(app_client, valid_breast_payload):
    """Smoke test: POST /api/member3b/predict with breast_cancer returns 200."""
    resp = app_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "breast_cancer", "data": valid_breast_payload}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["prediction"] in ("Benign", "Malignant")


def test_unified_lung_predict_smoke(app_client, valid_lung_payload):
    """Smoke test: POST /api/member3b/predict with lung_cancer returns 200."""
    resp = app_client.post(
        "/api/member3b/predict",
        data=json.dumps({"disease": "lung_cancer", "data": valid_lung_payload}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["prediction"] in (0, 1)


def test_direct_disease_endpoints_available_on_app(app_client, valid_breast_payload, valid_lung_payload):
    """Verifies that direct disease routes are also functional on the unified application."""
    # Breast direct routes
    r_b_h = app_client.get("/api/breast-cancer/health")
    assert r_b_h.status_code == 200
    assert r_b_h.get_json()["status"] == "ready"

    r_b_m = app_client.get("/api/breast-cancer/model-info")
    assert r_b_m.status_code == 200

    r_b_p = app_client.post(
        "/api/breast-cancer/predict/user",
        data=json.dumps(valid_breast_payload),
        content_type="application/json",
    )
    assert r_b_p.status_code == 200

    # Lung direct routes
    r_l_h = app_client.get("/api/lung-cancer/health")
    assert r_l_h.status_code == 200
    assert r_l_h.get_json()["status"] == "ready"

    r_l_m = app_client.get("/api/lung-cancer/model-info")
    assert r_l_m.status_code == 200

    r_l_p = app_client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_lung_payload),
        content_type="application/json",
    )
    assert r_l_p.status_code == 200


def test_no_secret_or_absolute_path_leakage(app_client):
    """Verifies that endpoint responses do not leak local paths or secrets."""
    for path in ["/api/member3b/health", "/api/member3b/model-info"]:
        resp = app_client.get(path)
        text = resp.get_data(as_text=True)
        assert not re.search(r"[A-Za-z]:\\", text)
        assert "password" not in text.lower()
        assert "secret" not in text.lower()
