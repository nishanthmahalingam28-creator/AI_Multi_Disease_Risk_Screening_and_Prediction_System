"""
Unit tests for the Lung Cancer Prediction Service Layer.

Validates:
- Service initialization and capability readiness
- Health check functionality and safety (no path leaks)
- Model metadata retrieval and fidelity
- User and clinical prediction workflows
- Cross-interface consistency between predict_user and predict_clinical
- Strict validation error propagation and encapsulation
- Absence of tracebacks, filesystem paths, or internal implementation leaks
- Verification of delegation to predict.py (no duplicate ML inference)
"""

from pathlib import Path
import pytest
import pandas as pd

from member3b.lung_cancer.src.service import (
    LungCancerPredictionService,
    DISEASE_KEY,
    SERVICE_NAME,
)
from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data
import member3b.lung_cancer.src.predict as predict_module


@pytest.fixture
def service():
    """Provides an initialized LungCancerPredictionService instance."""
    return LungCancerPredictionService()


@pytest.fixture
def valid_sample_input():
    """Provides a valid 15-feature dictionary extracted from raw data."""
    raw_df = load_raw_lung_cancer_data()
    row = raw_df.iloc[0]
    return {
        "gender": row["GENDER"],
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


# =========================================================================
# Service Initialization (Tests 1-4)
# =========================================================================

def test_1_service_imports_successfully():
    """Test 1: Verifies LungCancerPredictionService can be imported."""
    from member3b.lung_cancer.src.service import LungCancerPredictionService as ServiceClass
    assert ServiceClass is not None


def test_2_service_initializes_successfully(service):
    """Test 2: Verifies service instantiates cleanly."""
    assert service is not None
    assert isinstance(service, LungCancerPredictionService)


def test_3_model_artifact_is_available(service):
    """Test 3: Verifies model artifact is discoverable by the service."""
    service._ensure_artifacts()
    assert service._model is not None
    assert hasattr(service._model, "predict")


def test_4_preprocessor_artifact_is_available(service):
    """Test 4: Verifies preprocessor artifact is discoverable by the service."""
    service._ensure_artifacts()
    assert service._preprocessor is not None
    assert hasattr(service._preprocessor, "transform")


# =========================================================================
# Health Check (Tests 5-9)
# =========================================================================

def test_5_health_check_returns_dict(service):
    """Test 5: Verifies health_check returns a dictionary."""
    health = service.health_check()
    assert isinstance(health, dict)


def test_6_healthy_status_when_artifacts_available(service):
    """Test 6: Verifies health status is 'ready' when artifacts exist."""
    health = service.health_check()
    assert health["status"] == "ready"
    assert health["service"] == SERVICE_NAME
    assert health["disease"] == DISEASE_KEY


def test_7_model_loaded_status_is_correct(service):
    """Test 7: Verifies model_loaded flag is True."""
    health = service.health_check()
    assert health["model_loaded"] is True


def test_8_preprocessor_loaded_status_is_correct(service):
    """Test 8: Verifies preprocessor_loaded flag is True."""
    health = service.health_check()
    assert health["preprocessor_loaded"] is True


def test_9_no_filesystem_path_leaked_in_health_check(service):
    """Test 9: Verifies health_check exposes no absolute filesystem paths."""
    health = service.health_check()
    for k, v in health.items():
        v_str = str(v)
        assert ":\\" not in v_str, f"Leaked Windows path in key {k}: {v_str}"
        assert "/Users/" not in v_str, f"Leaked path in key {k}: {v_str}"


# =========================================================================
# Model Information (Tests 10-15)
# =========================================================================

def test_10_get_model_info_succeeds(service):
    """Test 10: Verifies get_model_info returns status 'success'."""
    info = service.get_model_info()
    assert info["status"] == "success"
    assert info["service"] == SERVICE_NAME
    assert info["disease"] == DISEASE_KEY


def test_11_model_name_is_correct(service):
    """Test 11: Verifies model name matches LogisticRegression."""
    info = service.get_model_info()
    assert "LogisticRegression" in info["model"]


def test_12_feature_count_is_15(service):
    """Test 12: Verifies feature count is exactly 15."""
    info = service.get_model_info()
    assert info["feature_count"] == 15
    assert len(info["features"]) == 15


def test_13_threshold_is_0_5(service):
    """Test 13: Verifies classification decision threshold is 0.50."""
    info = service.get_model_info()
    assert info["threshold"] == 0.50


def test_14_target_mapping_is_correct(service):
    """Test 14: Verifies target mapping is NO: 0, YES: 1."""
    info = service.get_model_info()
    assert info["target_mapping"] == {"NO": 0, "YES": 1}
    assert info["class_mapping"] == {0: "NO", 1: "YES"}


def test_15_no_internal_paths_exposed_in_model_info(service):
    """Test 15: Verifies model info response contains no filesystem paths."""
    info = service.get_model_info()
    for k, v in info.items():
        v_str = str(v)
        assert ":\\" not in v_str, f"Leaked path in key {k}: {v_str}"
        assert "/Users/" not in v_str, f"Leaked path in key {k}: {v_str}"


# =========================================================================
# User Prediction (Tests 16-21)
# =========================================================================

def test_16_valid_input_produces_successful_user_prediction(service, valid_sample_input):
    """Test 16: Verifies valid input yields status == 'success' in predict_user."""
    res = service.predict_user(valid_sample_input)
    assert res["status"] == "success"
    assert res["service"] == SERVICE_NAME
    assert res["disease"] == DISEASE_KEY


def test_17_user_prediction_is_0_or_1(service, valid_sample_input):
    """Test 17: Verifies prediction is either discrete 0 or 1."""
    res = service.predict_user(valid_sample_input)
    assert res["prediction"] in [0, 1]


def test_18_user_prediction_class_is_no_or_yes(service, valid_sample_input):
    """Test 18: Verifies prediction_class is 'NO' or 'YES'."""
    res = service.predict_user(valid_sample_input)
    assert res["prediction_class"] in ["NO", "YES"]


def test_19_user_probability_is_between_0_and_1(service, valid_sample_input):
    """Test 19: Verifies probability is bounded in [0.0, 1.0]."""
    res = service.predict_user(valid_sample_input)
    assert isinstance(res["probability"], float)
    assert 0.0 <= res["probability"] <= 1.0


def test_20_user_risk_percentage_is_between_0_and_100(service, valid_sample_input):
    """Test 20: Verifies risk percentage is bounded in [0.0, 100.0]."""
    res = service.predict_user(valid_sample_input)
    assert isinstance(res["risk_percentage"], (int, float))
    assert 0.0 <= res["risk_percentage"] <= 100.0


def test_21_service_result_matches_direct_predict_user(service, valid_sample_input):
    """Test 21: Verifies service predict_user matches predict.py output."""
    service_res = service.predict_user(valid_sample_input)
    direct_res = predict_module.predict_user(valid_sample_input)

    assert service_res["prediction"] == direct_res["prediction"]
    assert service_res["prediction_class"] == direct_res["prediction_class"]
    assert abs(service_res["probability"] - direct_res["probability"]) < 1e-6
    assert abs(service_res["risk_percentage"] - direct_res["risk_percentage"]) < 1e-4


# =========================================================================
# Clinical Prediction (Tests 22-23)
# =========================================================================

def test_22_valid_input_produces_successful_clinical_prediction(service, valid_sample_input):
    """Test 22: Verifies valid input yields status == 'success' in predict_clinical."""
    res = service.predict_clinical(valid_sample_input)
    assert res["status"] == "success"
    assert res["service"] == SERVICE_NAME
    assert res["prediction"] in [0, 1]
    assert res["prediction_class"] in ["NO", "YES"]


def test_23_service_result_matches_direct_predict_clinical(service, valid_sample_input):
    """Test 23: Verifies service predict_clinical matches predict.py output."""
    service_res = service.predict_clinical(valid_sample_input)
    direct_res = predict_module.predict_clinical(valid_sample_input)

    assert service_res["prediction"] == direct_res["prediction"]
    assert service_res["prediction_class"] == direct_res["prediction_class"]
    assert abs(service_res["probability"] - direct_res["probability"]) < 1e-6


# =========================================================================
# Cross-Interface Consistency (Tests 24-26)
# =========================================================================

def test_24_predict_user_and_clinical_same_prediction(service, valid_sample_input):
    """Test 24: Verifies predict_user and predict_clinical yield same prediction label."""
    user_res = service.predict_user(valid_sample_input)
    clin_res = service.predict_clinical(valid_sample_input)
    assert user_res["prediction"] == clin_res["prediction"]
    assert user_res["prediction_class"] == clin_res["prediction_class"]


def test_25_predict_user_and_clinical_same_probability(service, valid_sample_input):
    """Test 25: Verifies predict_user and predict_clinical yield identical probability."""
    user_res = service.predict_user(valid_sample_input)
    clin_res = service.predict_clinical(valid_sample_input)
    assert abs(user_res["probability"] - clin_res["probability"]) < 1e-6


def test_26_predict_user_and_clinical_same_risk_percentage(service, valid_sample_input):
    """Test 26: Verifies predict_user and predict_clinical yield identical risk percentage."""
    user_res = service.predict_user(valid_sample_input)
    clin_res = service.predict_clinical(valid_sample_input)
    assert abs(user_res["risk_percentage"] - clin_res["risk_percentage"]) < 1e-4


# =========================================================================
# Validation and Error Handling (Tests 27-33)
# =========================================================================

def test_27_missing_input_is_handled(service, valid_sample_input):
    """Test 27: Verifies missing feature returns structured validation error."""
    corrupt = valid_sample_input.copy()
    del corrupt["age"]
    res = service.predict_user(corrupt)
    assert res["status"] == "error"
    assert res["error_type"] == "validation_error"
    assert "Missing required feature: age" in res["message"]


def test_28_invalid_input_is_handled(service):
    """Test 28: Verifies None or malformed container returns validation error."""
    res_none = service.predict_user(None)
    assert res_none["status"] == "error"
    assert res_none["error_type"] == "validation_error"

    res_empty_df = service.predict_user(pd.DataFrame())
    assert res_empty_df["status"] == "error"


def test_29_invalid_binary_feature_is_handled(service, valid_sample_input):
    """Test 29: Verifies non-binary symptom entry returns structured validation error."""
    corrupt = valid_sample_input.copy()
    corrupt["smoking"] = 99
    res = service.predict_user(corrupt)
    assert res["status"] == "error"
    assert res["error_type"] == "validation_error"
    assert "Invalid value for smoking" in res["message"]


def test_30_invalid_gender_is_handled(service, valid_sample_input):
    """Test 30: Verifies invalid gender returns structured validation error."""
    corrupt = valid_sample_input.copy()
    corrupt["gender"] = "NON_BINARY"
    res = service.predict_user(corrupt)
    assert res["status"] == "error"
    assert res["error_type"] == "validation_error"
    assert "Invalid value for gender" in res["message"]


def test_31_nan_and_infinite_values_are_handled(service, valid_sample_input):
    """Test 31: Verifies NaN and Inf inputs return structured validation error."""
    corrupt_nan = valid_sample_input.copy()
    corrupt_nan["age"] = float("nan")
    res_nan = service.predict_user(corrupt_nan)
    assert res_nan["status"] == "error"
    assert "cannot be NaN" in res_nan["message"]

    corrupt_inf = valid_sample_input.copy()
    corrupt_inf["age"] = float("inf")
    res_inf = service.predict_user(corrupt_inf)
    assert res_inf["status"] == "error"
    assert "cannot be infinite" in res_inf["message"]


def test_32_error_response_does_not_expose_traceback(service, valid_sample_input):
    """Test 32: Verifies error messages do not leak Python stack traces."""
    corrupt = valid_sample_input.copy()
    corrupt["age"] = "invalid_number"
    res = service.predict_user(corrupt)
    assert res["status"] == "error"
    msg = res["message"]
    assert "Traceback" not in msg
    assert "File \"" not in msg
    assert "line " not in msg


def test_33_error_response_does_not_expose_filesystem_path(service, valid_sample_input):
    """Test 33: Verifies error responses do not leak local filesystem paths."""
    corrupt = valid_sample_input.copy()
    corrupt["age"] = -1
    res = service.predict_user(corrupt)
    assert res["status"] == "error"
    msg = res["message"]
    assert ":\\" not in msg
    assert "/Users/" not in msg


# =========================================================================
# Delegation Verification (Test 34)
# =========================================================================

def test_34_service_delegates_to_prediction_engine(service, valid_sample_input, monkeypatch):
    """Test 34: Verifies service delegates inference to predict.py instead of duplicate ML logic."""
    delegation_tracker = {"user_called": False, "clinical_called": False}

    import member3b.lung_cancer.src.service as s_mod

    orig_user = s_mod._predict_user_core
    orig_clinical = s_mod._predict_clinical_core

    def mock_predict_user(input_data, **kwargs):
        delegation_tracker["user_called"] = True
        return orig_user(input_data, **kwargs)

    def mock_predict_clinical(input_data, **kwargs):
        delegation_tracker["clinical_called"] = True
        return orig_clinical(input_data, **kwargs)

    monkeypatch.setattr(s_mod, "_predict_user_core", mock_predict_user)
    monkeypatch.setattr(s_mod, "_predict_clinical_core", mock_predict_clinical)

    res_u = service.predict_user(valid_sample_input)
    assert delegation_tracker["user_called"] is True
    assert res_u["status"] == "success"

    res_c = service.predict_clinical(valid_sample_input)
    assert delegation_tracker["clinical_called"] is True
    assert res_c["status"] == "success"


# =========================================================================
# Unavailable Artifacts Handling (Test 35)
# =========================================================================

def test_35_unavailable_artifacts_handling():
    """Test 35: Verifies service returns controlled failure when artifacts are missing."""
    empty_dir_service = LungCancerPredictionService(models_dir=Path("/non/existent/models/dir"))
    health = empty_dir_service.health_check()
    assert health["status"] == "unavailable"
    assert health["model_loaded"] is False
    assert health["preprocessor_loaded"] is False
