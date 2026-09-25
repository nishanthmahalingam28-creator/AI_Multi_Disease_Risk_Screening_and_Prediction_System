"""
Unit tests for Member 3B Unified Cancer Prediction Service Layer.

Verifies:
- Service orchestration across Breast Cancer and Lung Cancer prediction modules
- Health check aggregation and partial-failure reporting
- Safe model metadata aggregation
- Disease routing and envelope validation
- Strict delegation without logic duplication
"""

from unittest.mock import MagicMock
import pytest

from member3b.common.service import (
    CancerPredictionService,
    UNIFIED_SERVICE_NAME,
    SUPPORTED_DISEASES,
)
from member3b.breast_cancer.src.service import BreastCancerPredictionService
from member3b.lung_cancer.src.service import LungCancerPredictionService
from member3b.breast_cancer.src.data_loader import load_raw_breast_cancer_data
from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data


@pytest.fixture
def service():
    """Provides a live CancerPredictionService instance."""
    return CancerPredictionService()


@pytest.fixture
def valid_breast_payload():
    """Provides valid 30-feature Breast Cancer payload."""
    df = load_raw_breast_cancer_data()
    feature_cols = [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
    return {col: float(df.iloc[0][col]) for col in feature_cols}


@pytest.fixture
def valid_lung_payload():
    """Provides valid 15-feature Lung Cancer payload."""
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
# 1. Initialization and Dependency Injection Tests
# =============================================================================

def test_service_init_default(service):
    """Verifies default service initializes both underlying disease services."""
    assert isinstance(service.breast_service, BreastCancerPredictionService)
    assert isinstance(service.lung_service, LungCancerPredictionService)


def test_service_init_injected():
    """Verifies service accepts injected mock or custom disease services."""
    mock_b = MagicMock(spec=BreastCancerPredictionService)
    mock_l = MagicMock(spec=LungCancerPredictionService)
    custom_service = CancerPredictionService(breast_service=mock_b, lung_service=mock_l)
    assert custom_service.breast_service is mock_b
    assert custom_service.lung_service is mock_l


# =============================================================================
# 2. Health Check Aggregation Tests
# =============================================================================

def test_health_check_both_ready(service):
    """Verifies health check reports 'ready' when both sub-services are ready."""
    health = service.health_check()
    assert health["status"] == "ready"
    assert health["service"] == UNIFIED_SERVICE_NAME
    assert "breast_cancer" in health["diseases"]
    assert "lung_cancer" in health["diseases"]
    assert health["diseases"]["breast_cancer"]["status"] == "ready"
    assert health["diseases"]["lung_cancer"]["status"] == "ready"


def test_health_check_breast_unavailable():
    """Verifies health check reports 'unavailable' if Breast Cancer service is unhealthy."""
    mock_b = MagicMock(spec=BreastCancerPredictionService)
    mock_b.health_check.return_value = {"status": "unhealthy", "disease": "breast_cancer"}
    mock_l = MagicMock(spec=LungCancerPredictionService)
    mock_l.health_check.return_value = {"status": "ready", "disease": "lung_cancer"}

    svc = CancerPredictionService(breast_service=mock_b, lung_service=mock_l)
    health = svc.health_check()

    assert health["status"] == "unavailable"
    assert health["diseases"]["breast_cancer"]["status"] == "unhealthy"
    assert health["diseases"]["lung_cancer"]["status"] == "ready"


def test_health_check_lung_unavailable():
    """Verifies health check reports 'unavailable' if Lung Cancer service is unavailable."""
    mock_b = MagicMock(spec=BreastCancerPredictionService)
    mock_b.health_check.return_value = {"status": "ready", "disease": "breast_cancer"}
    mock_l = MagicMock(spec=LungCancerPredictionService)
    mock_l.health_check.return_value = {"status": "unavailable", "disease": "lung_cancer"}

    svc = CancerPredictionService(breast_service=mock_b, lung_service=mock_l)
    health = svc.health_check()

    assert health["status"] == "unavailable"
    assert health["diseases"]["breast_cancer"]["status"] == "ready"
    assert health["diseases"]["lung_cancer"]["status"] == "unavailable"


def test_health_check_exception_handling():
    """Verifies health check catches unhandled exceptions and reports 'unavailable'."""
    mock_b = MagicMock(spec=BreastCancerPredictionService)
    mock_b.health_check.side_effect = RuntimeError("Fatal crash")
    svc = CancerPredictionService(breast_service=mock_b)
    health = svc.health_check()
    assert health["status"] == "unavailable"


# =============================================================================
# 3. Model Information Aggregation Tests
# =============================================================================

def test_get_model_info_success(service):
    """Verifies model metadata aggregation returns status success and both model profiles."""
    info = service.get_model_info()
    assert info["status"] == "success"
    assert info["service"] == UNIFIED_SERVICE_NAME
    assert "models" in info
    assert "breast_cancer" in info["models"]
    assert "lung_cancer" in info["models"]

    b_info = info["models"]["breast_cancer"]
    l_info = info["models"]["lung_cancer"]
    assert b_info["disease"] == "breast_cancer"
    assert l_info["disease"] == "lung_cancer"
    assert b_info["feature_count"] == 30
    assert l_info["feature_count"] == 15


def test_get_model_info_failure_handling():
    """Verifies get_model_info returns 'unavailable' when a sub-service fails."""
    mock_b = MagicMock(spec=BreastCancerPredictionService)
    mock_b.get_model_info.side_effect = RuntimeError("Disk IO failure")
    svc = CancerPredictionService(breast_service=mock_b)
    info = svc.get_model_info()
    assert info["status"] == "unavailable"
    assert "service_error" in info["error_type"]


# =============================================================================
# 4. Disease Routing and Prediction Execution Tests
# =============================================================================

def test_predict_breast_cancer_user(service, valid_breast_payload):
    """Verifies valid Breast Cancer screening prediction routes correctly in user mode."""
    res = service.predict(disease="breast_cancer", data=valid_breast_payload, mode="user")
    assert res["status"] == "success"
    assert res["disease"] == "breast_cancer"
    assert res["prediction"] in ("Benign", "Malignant")
    assert res["prediction_class"] in (0, 1)
    assert 0.0 <= res["probability"] <= 1.0


def test_predict_breast_cancer_clinical(service, valid_breast_payload):
    """Verifies valid Breast Cancer screening prediction routes correctly in clinical mode."""
    res = service.predict(disease="breast_cancer", data=valid_breast_payload, mode="clinical")
    assert res["status"] == "success"
    assert res["disease"] == "breast_cancer"
    assert res["prediction"] in ("Benign", "Malignant")


def test_predict_lung_cancer_user(service, valid_lung_payload):
    """Verifies valid Lung Cancer screening prediction routes correctly in user mode."""
    res = service.predict(disease="lung_cancer", data=valid_lung_payload, mode="user")
    assert res["status"] == "success"
    assert res["disease"] == "lung_cancer"
    assert res["prediction"] in (0, 1)
    assert res["prediction_class"] in ("NO", "YES")
    assert 0.0 <= res["probability"] <= 1.0


def test_predict_lung_cancer_clinical(service, valid_lung_payload):
    """Verifies valid Lung Cancer screening prediction routes correctly in clinical mode."""
    res = service.predict(disease="lung_cancer", data=valid_lung_payload, mode="clinical")
    assert res["status"] == "success"
    assert res["disease"] == "lung_cancer"
    assert res["prediction"] in (0, 1)


# =============================================================================
# 5. Envelope Validation Tests
# =============================================================================

def test_predict_invalid_disease_rejected(service, valid_lung_payload):
    """Verifies unsupported, empty, null, or non-string disease identifiers are rejected."""
    invalid_diseases = [
        None,
        "",
        "   ",
        "heart_cancer",
        "stroke",
        "diabetes",
        "unknown_disease",
        123,
    ]
    for d in invalid_diseases:
        res = service.predict(disease=d, data=valid_lung_payload)
        assert res["status"] == "error"
        assert res["error_type"] == "validation_error"


def test_predict_invalid_data_container_rejected(service):
    """Verifies missing, non-dict, or empty data containers are rejected at envelope stage."""
    invalid_containers = [
        None,
        [],
        "string_data",
        12345,
        True,
        {},
    ]
    for c in invalid_containers:
        res = service.predict(disease="lung_cancer", data=c)
        assert res["status"] == "error"
        assert res["error_type"] == "validation_error"


def test_predict_invalid_mode_rejected(service, valid_lung_payload):
    """Verifies unsupported prediction mode returns validation error."""
    res = service.predict(disease="lung_cancer", data=valid_lung_payload, mode="unsupported_mode")
    assert res["status"] == "error"
    assert res["error_type"] == "validation_error"
    assert "Invalid mode" in res["message"]


def test_predict_delegation_without_duplication(valid_lung_payload):
    """Verifies that unified service strictly delegates to underlying service without re-implementing ML."""
    mock_lung = MagicMock(spec=LungCancerPredictionService)
    mock_lung.predict_user.return_value = {
        "status": "success",
        "disease": "lung_cancer",
        "prediction": 1,
        "prediction_class": "YES",
        "probability": 0.9,
        "risk_percentage": 90.0,
        "model": "LogisticRegression",
        "threshold": 0.5,
    }
    svc = CancerPredictionService(lung_service=mock_lung)
    res = svc.predict(disease="lung_cancer", data=valid_lung_payload, mode="user")
    assert res["status"] == "success"
    mock_lung.predict_user.assert_called_once_with(valid_lung_payload)
