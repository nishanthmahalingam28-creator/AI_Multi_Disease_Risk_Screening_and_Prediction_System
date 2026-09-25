"""
Unit and integration tests for the Lung Cancer Flask REST API Layer.

Covers at minimum:
Health:
1. GET health returns 200.
2. Response is JSON.
3. Status is ready.
4. Model-loaded status is correct.
5. Preprocessor-loaded status is correct.

Model info:
6. GET model-info returns 200.
7. Response is JSON.
8. Model is LogisticRegression.
9. Feature count is 15.
10. Threshold is 0.5.
11. Target mapping is correct.
12. No filesystem path is exposed.

User prediction:
13. Valid POST returns 200.
14. Response is JSON.
15. Prediction is 0 or 1.
16. Prediction class is NO or YES.
17. Probability is 0–1.
18. Risk percentage is 0–100.
19. Response contains model and threshold.
20. API result matches the service result.

Clinical prediction:
21. Valid POST returns 200.
22. Response is JSON.
23. API result matches service result.

Cross-interface consistency:
24. User and clinical API calls with identical input produce identical prediction.
25. Probability is identical.
26. Risk percentage is identical.

Invalid requests:
27. Empty POST body returns 400.
28. Malformed JSON returns 400.
29. JSON array returns 400.
30. JSON string returns 400.
31. JSON null returns 400.
32. Missing required feature returns 400.
33. Unexpected feature returns 400.
34. Invalid gender returns 400.
35. Invalid binary value returns 400.
36. NaN is rejected.
37. Infinity is rejected.
38. Duplicate JSON key is rejected.

HTTP behavior:
39. Wrong method returns 405.
40. Unknown endpoint returns 404.
41. Error responses are JSON.
42. Error responses do not expose traceback.
43. Error responses do not expose filesystem paths.

Service delegation:
44. Verify API routes delegate to LungCancerPredictionService.

Additional tests:
45. Health endpoint returns 503 when service is unavailable.
46. Non-JSON Content-Type returns 400.
47. Service error returns 503 on prediction.
"""

from typing import Dict, Any
import json
import re
from unittest.mock import MagicMock, patch
import pytest

from member3b.lung_cancer.src.api import create_app
from member3b.lung_cancer.src.service import LungCancerPredictionService, SERVICE_NAME, DISEASE_KEY


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
def valid_payload() -> Dict[str, Any]:
    """Provides a valid 15-feature Lung Cancer survey payload."""
    return {
        "gender": "MALE",
        "age": 69,
        "smoking": 0,
        "yellow_fingers": 1,
        "anxiety": 1,
        "peer_pressure": 0,
        "chronic_disease": 0,
        "fatigue": 1,
        "allergy": 0,
        "wheezing": 1,
        "alcohol_consuming": 1,
        "coughing": 1,
        "shortness_of_breath": 1,
        "swallowing_difficulty": 1,
        "chest_pain": 1,
    }


# ==============================================================================
# Health Endpoint Tests (1 - 5)
# ==============================================================================

def test_1_get_health_returns_200(client):
    """Verify GET /api/lung-cancer/health returns HTTP 200."""
    response = client.get("/api/lung-cancer/health")
    assert response.status_code == 200


def test_2_health_response_is_json(client):
    """Verify GET /api/lung-cancer/health returns application/json Content-Type."""
    response = client.get("/api/lung-cancer/health")
    assert response.is_json
    assert "application/json" in response.content_type


def test_3_health_status_is_ready(client):
    """Verify GET /api/lung-cancer/health returns status ready."""
    data = client.get("/api/lung-cancer/health").get_json()
    assert data["status"] == "ready"
    assert data["service"] == SERVICE_NAME


def test_4_health_model_loaded_is_true(client):
    """Verify GET /api/lung-cancer/health indicates model_loaded is True."""
    data = client.get("/api/lung-cancer/health").get_json()
    assert data["model_loaded"] is True


def test_5_health_preprocessor_loaded_is_true(client):
    """Verify GET /api/lung-cancer/health indicates preprocessor_loaded is True."""
    data = client.get("/api/lung-cancer/health").get_json()
    assert data["preprocessor_loaded"] is True


# ==============================================================================
# Model Info Endpoint Tests (6 - 12)
# ==============================================================================

def test_6_get_model_info_returns_200(client):
    """Verify GET /api/lung-cancer/model-info returns HTTP 200."""
    response = client.get("/api/lung-cancer/model-info")
    assert response.status_code == 200


def test_7_model_info_response_is_json(client):
    """Verify GET /api/lung-cancer/model-info returns valid JSON."""
    response = client.get("/api/lung-cancer/model-info")
    assert response.is_json
    data = response.get_json()
    assert isinstance(data, dict)
    assert data["status"] == "success"


def test_8_model_info_model_is_logistic_regression(client):
    """Verify GET /api/lung-cancer/model-info reports LogisticRegression."""
    data = client.get("/api/lung-cancer/model-info").get_json()
    assert data["model"] == "LogisticRegression"


def test_9_model_info_feature_count_is_15(client):
    """Verify GET /api/lung-cancer/model-info reports exactly 15 features."""
    data = client.get("/api/lung-cancer/model-info").get_json()
    assert data["feature_count"] == 15
    if "features" in data:
        assert len(data["features"]) == 15


def test_10_model_info_threshold_is_0_5(client):
    """Verify GET /api/lung-cancer/model-info reports 0.5 decision threshold."""
    data = client.get("/api/lung-cancer/model-info").get_json()
    assert abs(data["threshold"] - 0.5) < 1e-6


def test_11_model_info_target_mapping_is_correct(client):
    """Verify GET /api/lung-cancer/model-info reports correct target mapping."""
    data = client.get("/api/lung-cancer/model-info").get_json()
    assert data["target_mapping"] == {"NO": 0, "YES": 1}


def test_12_model_info_no_filesystem_paths_exposed(client):
    """Verify GET /api/lung-cancer/model-info does not leak filesystem paths."""
    text = client.get("/api/lung-cancer/model-info").get_data(as_text=True)
    assert not re.search(r"[A-Za-z]:\\", text)
    assert "/Users/" not in text
    assert "/home/" not in text


# ==============================================================================
# User Prediction Tests (13 - 20)
# ==============================================================================

def test_13_valid_user_post_returns_200(client, valid_payload):
    """Verify POST /api/lung-cancer/predict/user returns HTTP 200."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    )
    assert response.status_code == 200


def test_14_user_response_is_json(client, valid_payload):
    """Verify POST /api/lung-cancer/predict/user returns valid JSON with status success."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    )
    assert response.is_json
    data = response.get_json()
    assert data["status"] == "success"


def test_15_user_prediction_is_0_or_1(client, valid_payload):
    """Verify prediction field is either 0 or 1."""
    data = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert data["prediction"] in (0, 1)


def test_16_user_prediction_class_is_no_or_yes(client, valid_payload):
    """Verify prediction_class field is NO or YES."""
    data = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert data["prediction_class"] in ("NO", "YES")


def test_17_user_probability_between_0_and_1(client, valid_payload):
    """Verify probability field is between 0.0 and 1.0."""
    data = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert 0.0 <= data["probability"] <= 1.0


def test_18_user_risk_percentage_between_0_and_100(client, valid_payload):
    """Verify risk_percentage is between 0.0 and 100.0."""
    data = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert 0.0 <= data["risk_percentage"] <= 100.0
    assert abs(data["risk_percentage"] - (data["probability"] * 100)) < 1e-3


def test_19_user_response_contains_model_and_threshold(client, valid_payload):
    """Verify response includes model and threshold fields."""
    data = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert data["model"] == "LogisticRegression"
    assert abs(data["threshold"] - 0.5) < 1e-6
    assert "screening_interpretation" in data


def test_20_user_api_result_matches_service(client, valid_payload):
    """Verify API prediction response matches direct service call result."""
    service = LungCancerPredictionService()
    expected = service.predict_user(valid_payload)
    actual = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert actual["prediction"] == expected["prediction"]
    assert actual["prediction_class"] == expected["prediction_class"]
    assert abs(actual["probability"] - expected["probability"]) < 1e-6
    assert abs(actual["risk_percentage"] - expected["risk_percentage"]) < 1e-4


# ==============================================================================
# Clinical Prediction Tests (21 - 23)
# ==============================================================================

def test_21_valid_clinical_post_returns_200(client, valid_payload):
    """Verify POST /api/lung-cancer/predict/clinical returns HTTP 200."""
    response = client.post(
        "/api/lung-cancer/predict/clinical",
        data=json.dumps(valid_payload),
        content_type="application/json",
    )
    assert response.status_code == 200


def test_22_clinical_response_is_json(client, valid_payload):
    """Verify POST /api/lung-cancer/predict/clinical returns valid JSON with status success."""
    response = client.post(
        "/api/lung-cancer/predict/clinical",
        data=json.dumps(valid_payload),
        content_type="application/json",
    )
    assert response.is_json
    data = response.get_json()
    assert data["status"] == "success"


def test_23_clinical_api_result_matches_service(client, valid_payload):
    """Verify clinical API result matches direct service call."""
    service = LungCancerPredictionService()
    expected = service.predict_clinical(valid_payload)
    actual = client.post(
        "/api/lung-cancer/predict/clinical",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert actual["prediction"] == expected["prediction"]
    assert actual["prediction_class"] == expected["prediction_class"]
    assert abs(actual["probability"] - expected["probability"]) < 1e-6


# ==============================================================================
# Cross-Interface Consistency Tests (24 - 26)
# ==============================================================================

def test_24_cross_interface_same_prediction(client, valid_payload):
    """Verify user and clinical endpoints return identical binary prediction."""
    res_user = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    res_clin = client.post(
        "/api/lung-cancer/predict/clinical",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert res_user["prediction"] == res_clin["prediction"]
    assert res_user["prediction_class"] == res_clin["prediction_class"]


def test_25_cross_interface_same_probability(client, valid_payload):
    """Verify user and clinical endpoints return identical probabilities."""
    res_user = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    res_clin = client.post(
        "/api/lung-cancer/predict/clinical",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert abs(res_user["probability"] - res_clin["probability"]) < 1e-6


def test_26_cross_interface_same_risk_percentage(client, valid_payload):
    """Verify user and clinical endpoints return identical risk percentages."""
    res_user = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    res_clin = client.post(
        "/api/lung-cancer/predict/clinical",
        data=json.dumps(valid_payload),
        content_type="application/json",
    ).get_json()
    assert abs(res_user["risk_percentage"] - res_clin["risk_percentage"]) < 1e-4


# ==============================================================================
# Invalid Request Tests (27 - 38)
# ==============================================================================

def test_27_empty_post_body_returns_400(client):
    """Verify empty POST body returns HTTP 400."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data="",
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "invalid_request"


def test_28_malformed_json_returns_400(client):
    """Verify malformed JSON syntax returns HTTP 400."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data="{malformed json, age: 60",
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "invalid_request"


def test_29_json_array_returns_400(client):
    """Verify JSON array body returns HTTP 400."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps([{"age": 60}]),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "invalid_request"


def test_30_json_string_returns_400(client):
    """Verify JSON string body returns HTTP 400."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps("plain string payload"),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_31_json_null_returns_400(client):
    """Verify JSON null body returns HTTP 400."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data="null",
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_32_missing_required_feature_returns_400(client, valid_payload):
    """Verify missing required feature returns HTTP 400 with validation_error."""
    bad_payload = valid_payload.copy()
    del bad_payload["age"]
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(bad_payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Missing required feature" in data["message"]


def test_33_unexpected_feature_returns_400(client, valid_payload):
    """Verify unexpected extra feature returns HTTP 400."""
    bad_payload = valid_payload.copy()
    bad_payload["extra_unknown_feature"] = 123
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(bad_payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Unexpected feature" in data["message"]


def test_34_invalid_gender_returns_400(client, valid_payload):
    """Verify invalid gender string returns HTTP 400."""
    bad_payload = valid_payload.copy()
    bad_payload["gender"] = "UNKNOWN_GENDER"
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(bad_payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"


def test_35_invalid_binary_value_returns_400(client, valid_payload):
    """Verify non-binary symptom value (e.g. 5) returns HTTP 400."""
    bad_payload = valid_payload.copy()
    bad_payload["smoking"] = 5
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(bad_payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"


def test_36_nan_is_rejected_with_400(client, valid_payload):
    """Verify NaN value is rejected with HTTP 400."""
    bad_payload = valid_payload.copy()
    bad_payload["age"] = float("nan")
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(bad_payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "NaN" in data["message"]


def test_37_infinity_is_rejected_with_400(client, valid_payload):
    """Verify Infinity value is rejected with HTTP 400."""
    bad_payload = valid_payload.copy()
    bad_payload["age"] = float("inf")
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(bad_payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "infinite" in data["message"]


def test_38_duplicate_json_key_is_rejected_with_400(client):
    """Verify duplicate JSON keys in request payload are rejected with HTTP 400."""
    raw_json_with_duplicate = '{"gender": "MALE", "age": 55, "age": 65, "smoking": 1}'
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=raw_json_with_duplicate,
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "validation_error"
    assert "Duplicate JSON field: age" in data["message"]


# ==============================================================================
# HTTP Protocol Behavior Tests (39 - 43)
# ==============================================================================

def test_39_wrong_method_returns_405(client):
    """Verify GET on POST-only endpoint returns HTTP 405 Method Not Allowed."""
    response = client.get("/api/lung-cancer/predict/user")
    assert response.status_code == 405
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "method_not_allowed"


def test_40_unknown_endpoint_returns_404(client):
    """Verify unknown endpoint returns HTTP 404 with standardized JSON error."""
    response = client.get("/api/lung-cancer/non-existent-endpoint")
    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "not_found"


def test_41_error_responses_are_json(client):
    """Verify error responses have application/json content type."""
    response = client.get("/api/lung-cancer/non-existent-endpoint")
    assert response.is_json
    assert "application/json" in response.content_type


def test_42_error_responses_do_not_expose_traceback(client):
    """Verify error responses do not contain Traceback or internal exception text."""
    response = client.get("/api/lung-cancer/non-existent-endpoint")
    text = response.get_data(as_text=True)
    assert "Traceback" not in text
    assert "File \"" not in text


def test_43_error_responses_do_not_expose_filesystem_paths(client):
    """Verify error responses do not expose absolute filesystem paths."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data="invalid json",
        content_type="application/json",
    )
    text = response.get_data(as_text=True)
    assert not re.search(r"[A-Za-z]:\\", text)
    assert "/Users/" not in text
    assert "/home/" not in text


# ==============================================================================
# Service Delegation Tests (44 - 47)
# ==============================================================================

def test_44_service_delegation_via_mock():
    """Verify API prediction endpoints delegate directly to LungCancerPredictionService."""
    mock_service = MagicMock(spec=LungCancerPredictionService)
    mock_service.predict_user.return_value = {
        "status": "success",
        "service": SERVICE_NAME,
        "disease": DISEASE_KEY,
        "prediction": 1,
        "prediction_class": "YES",
        "probability": 0.85,
        "risk_percentage": 85.0,
        "model": "LogisticRegression",
        "threshold": 0.5,
        "screening_interpretation": "Higher predicted risk",
    }
    mock_service.predict_clinical.return_value = {
        "status": "success",
        "service": SERVICE_NAME,
        "disease": DISEASE_KEY,
        "prediction": 1,
        "prediction_class": "YES",
        "probability": 0.85,
        "risk_percentage": 85.0,
        "model": "LogisticRegression",
        "threshold": 0.5,
        "screening_interpretation": "Higher predicted risk",
    }

    mock_app = create_app(test_config={"TESTING": True}, service=mock_service)
    mock_client = mock_app.test_client()

    payload = {"age": 50}
    # Test predict_user delegation
    res_user = mock_client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res_user.status_code == 200
    mock_service.predict_user.assert_called_once_with(payload)

    # Test predict_clinical delegation
    res_clin = mock_client.post(
        "/api/lung-cancer/predict/clinical",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res_clin.status_code == 200
    mock_service.predict_clinical.assert_called_once_with(payload)


def test_45_health_check_returns_503_when_service_unavailable():
    """Verify GET /api/lung-cancer/health returns 503 if service reports unavailable."""
    mock_service = MagicMock(spec=LungCancerPredictionService)
    mock_service.health_check.return_value = {
        "status": "unavailable",
        "service": SERVICE_NAME,
        "disease": DISEASE_KEY,
        "model_loaded": False,
        "preprocessor_loaded": False,
    }

    mock_app = create_app(test_config={"TESTING": True}, service=mock_service)
    mock_client = mock_app.test_client()

    response = mock_client.get("/api/lung-cancer/health")
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "unavailable"


def test_46_non_json_content_type_returns_400(client, valid_payload):
    """Verify non-JSON Content-Type (e.g. text/plain) returns HTTP 400."""
    response = client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps(valid_payload),
        content_type="text/plain",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "invalid_request"


def test_47_service_error_returns_503_on_prediction():
    """Verify internal service error (e.g. missing artifacts) returns 503."""
    mock_service = MagicMock(spec=LungCancerPredictionService)
    mock_service.predict_user.return_value = {
        "status": "error",
        "service": SERVICE_NAME,
        "disease": DISEASE_KEY,
        "error_type": "service_error",
        "message": "Lung cancer prediction service is currently unavailable.",
    }

    mock_app = create_app(test_config={"TESTING": True}, service=mock_service)
    mock_client = mock_app.test_client()

    response = mock_client.post(
        "/api/lung-cancer/predict/user",
        data=json.dumps({"age": 50}),
        content_type="application/json",
    )
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_type"] == "service_error"
