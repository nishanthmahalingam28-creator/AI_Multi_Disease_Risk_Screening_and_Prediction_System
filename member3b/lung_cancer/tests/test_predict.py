"""
Lung Cancer Prediction Engine Tests.

Tests at minimum:
1. Prediction module imports successfully.
2. Model artifact loads.
3. Preprocessor artifact loads.
4. Valid numeric input works.
5. Valid YES/NO input works.
6. Valid lowercase gender works (case-insensitive).
7. Valid lowercase yes/no works.
8. Missing field is rejected.
9. Unexpected field is rejected.
10. Invalid gender is rejected.
11. Invalid binary value is rejected.
12. Age zero is rejected.
13. Negative age is rejected.
14. NaN is rejected.
15. Positive infinity is rejected.
16. Negative infinity is rejected.
17. Prediction is either 0 or 1.
18. prediction_class is either NO or YES.
19. Probability is between 0 and 1.
20. Risk percentage is between 0 and 100.
21. prediction corresponds correctly to the 0.5 threshold.
22. predict_user() and predict_clinical() return the same prediction for the same input.
23. predict_user() and predict_clinical() return the same probability for the same input.
24. Input feature order is deterministic.
25. Reloading the model gives consistent predictions.
26. Real dataset rows pass through the prediction engine.
27. Boolean literals (True/False) are explicitly rejected.
28. raise_on_error=True raises LungCancerValidationError.
"""

import math
import numpy as np
import pandas as pd
import pytest

from member3b.lung_cancer.src.data_loader import load_raw_lung_cancer_data
from member3b.lung_cancer.src.predict import (
    predict_user,
    predict_clinical,
    predict_lung_cancer,
    validate_input,
    load_artifacts,
    LungCancerValidationError,
    LungCancerModelError,
    EXPECTED_PREDICTOR_FEATURES,
    DEFAULT_THRESHOLD,
)


@pytest.fixture
def valid_numeric_input():
    """A valid dictionary with numeric 0/1 symptom inputs."""
    return {
        "gender": "MALE",
        "age": 69.0,
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


@pytest.fixture
def valid_string_input():
    """A valid dictionary with human-readable YES/NO inputs."""
    return {
        "gender": "FEMALE",
        "age": 59.0,
        "smoking": "NO",
        "yellow_fingers": "NO",
        "anxiety": "NO",
        "peer_pressure": "YES",
        "chronic_disease": "NO",
        "fatigue": "YES",
        "allergy": "NO",
        "wheezing": "YES",
        "alcohol_consuming": "NO",
        "coughing": "YES",
        "shortness_of_breath": "YES",
        "swallowing_difficulty": "NO",
        "chest_pain": "YES",
    }


# -------------------------------------------------------------------------
# Test 1: Prediction module imports successfully
# -------------------------------------------------------------------------
def test_1_prediction_module_imports():
    """Test 1: Verifies module imports cleanly with all required interfaces."""
    import member3b.lung_cancer.src.predict as predict_module
    assert hasattr(predict_module, "predict_user")
    assert hasattr(predict_module, "predict_clinical")
    assert hasattr(predict_module, "validate_input")
    assert hasattr(predict_module, "load_artifacts")


# -------------------------------------------------------------------------
# Test 2: Model artifact loads
# -------------------------------------------------------------------------
def test_2_model_artifact_loads():
    """Test 2: Verifies model artifact loads and has prediction capabilities."""
    _, model, _ = load_artifacts()
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


# -------------------------------------------------------------------------
# Test 3: Preprocessor artifact loads
# -------------------------------------------------------------------------
def test_3_preprocessor_artifact_loads():
    """Test 3: Verifies preprocessor artifact loads and exposes transform."""
    preprocessor, _, _ = load_artifacts()
    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")


# -------------------------------------------------------------------------
# Test 4: Valid numeric input works
# -------------------------------------------------------------------------
def test_4_valid_numeric_input_works(valid_numeric_input):
    """Test 4: Verifies valid numeric dictionary succeeds."""
    res = predict_user(valid_numeric_input)
    assert res["status"] == "success"
    assert res["prediction"] in [0, 1]
    assert res["prediction_class"] in ["NO", "YES"]
    assert 0.0 <= res["probability"] <= 1.0


# -------------------------------------------------------------------------
# Test 5: Valid YES/NO input works
# -------------------------------------------------------------------------
def test_5_valid_yes_no_input_works(valid_string_input):
    """Test 5: Verifies human-readable YES/NO input succeeds."""
    res = predict_user(valid_string_input)
    assert res["status"] == "success"
    assert res["prediction"] in [0, 1]
    assert res["prediction_class"] in ["NO", "YES"]


# -------------------------------------------------------------------------
# Test 6: Valid lowercase gender works if case-insensitive
# -------------------------------------------------------------------------
def test_6_valid_lowercase_gender_works(valid_numeric_input):
    """Test 6: Verifies case-insensitive gender handling."""
    data_male = valid_numeric_input.copy()
    data_male["gender"] = "male"
    res_m = predict_user(data_male)
    assert res_m["status"] == "success"

    data_female = valid_numeric_input.copy()
    data_female["gender"] = "female"
    res_f = predict_user(data_female)
    assert res_f["status"] == "success"


# -------------------------------------------------------------------------
# Test 7: Valid lowercase yes/no works if supported
# -------------------------------------------------------------------------
def test_7_valid_lowercase_yes_no_works(valid_numeric_input):
    """Test 7: Verifies case-insensitive yes/no symptom values."""
    data = valid_numeric_input.copy()
    data["smoking"] = "yes"
    data["fatigue"] = "no"
    res = predict_user(data)
    assert res["status"] == "success"


# -------------------------------------------------------------------------
# Test 8: Missing field is rejected
# -------------------------------------------------------------------------
def test_8_missing_field_is_rejected(valid_numeric_input):
    """Test 8: Rejects dictionary missing required fields."""
    corrupt = valid_numeric_input.copy()
    del corrupt["age"]
    res = predict_user(corrupt)
    assert res["status"] == "error"
    assert "Missing required feature: age" in res["message"]

    with pytest.raises(LungCancerValidationError, match="Missing required feature: age"):
        predict_user(corrupt, raise_on_error=True)


# -------------------------------------------------------------------------
# Test 9: Unexpected field is rejected
# -------------------------------------------------------------------------
def test_9_unexpected_field_is_rejected(valid_numeric_input):
    """Test 9: Rejects inputs with unknown extraneous fields."""
    corrupt = valid_numeric_input.copy()
    corrupt["blood_pressure"] = 120
    res = predict_user(corrupt)
    assert res["status"] == "error"
    assert "Unexpected feature encountered: blood_pressure" in res["message"]

    with pytest.raises(LungCancerValidationError, match="Unexpected feature encountered"):
        predict_user(corrupt, raise_on_error=True)


# -------------------------------------------------------------------------
# Test 10: Invalid gender is rejected
# -------------------------------------------------------------------------
def test_10_invalid_gender_is_rejected(valid_numeric_input):
    """Test 10: Rejects invalid or out-of-domain gender entries."""
    for bad_gender in ["OTHER", "UNKNOWN", "M", "F", 2, -1]:
        corrupt = valid_numeric_input.copy()
        corrupt["gender"] = bad_gender
        res = predict_user(corrupt)
        assert res["status"] == "error", f"Failed to reject gender: {bad_gender}"
        assert "Invalid" in res["message"] or "Expected" in res["message"]


# -------------------------------------------------------------------------
# Test 11: Invalid binary value is rejected
# -------------------------------------------------------------------------
def test_11_invalid_binary_value_is_rejected(valid_numeric_input):
    """Test 11: Rejects symptom values not in {0, 1, YES, NO}."""
    for bad_val in [2, -1, "maybe", "unknown", "", "TRUE", "FALSE"]:
        corrupt = valid_numeric_input.copy()
        corrupt["smoking"] = bad_val
        res = predict_user(corrupt)
        assert res["status"] == "error", f"Failed to reject binary: {bad_val}"
        assert "Invalid value for smoking" in res["message"]


# -------------------------------------------------------------------------
# Test 12: Age zero is rejected
# -------------------------------------------------------------------------
def test_12_age_zero_is_rejected(valid_numeric_input):
    """Test 12: Rejects age equal to zero."""
    corrupt = valid_numeric_input.copy()
    corrupt["age"] = 0
    res = predict_user(corrupt)
    assert res["status"] == "error"
    assert "Age must be positive" in res["message"]


# -------------------------------------------------------------------------
# Test 13: Negative age is rejected
# -------------------------------------------------------------------------
def test_13_negative_age_is_rejected(valid_numeric_input):
    """Test 13: Rejects negative age."""
    corrupt = valid_numeric_input.copy()
    corrupt["age"] = -45
    res = predict_user(corrupt)
    assert res["status"] == "error"
    assert "Age must be positive" in res["message"]


# -------------------------------------------------------------------------
# Test 14: NaN is rejected
# -------------------------------------------------------------------------
def test_14_nan_is_rejected(valid_numeric_input):
    """Test 14: Rejects NaN in age or symptom fields."""
    corrupt_age = valid_numeric_input.copy()
    corrupt_age["age"] = float("nan")
    res_age = predict_user(corrupt_age)
    assert res_age["status"] == "error"
    assert "cannot be NaN" in res_age["message"]

    corrupt_sym = valid_numeric_input.copy()
    corrupt_sym["coughing"] = float("nan")
    res_sym = predict_user(corrupt_sym)
    assert res_sym["status"] == "error"
    assert "cannot be NaN" in res_sym["message"]


# -------------------------------------------------------------------------
# Test 15: Positive infinity is rejected
# -------------------------------------------------------------------------
def test_15_positive_infinity_is_rejected(valid_numeric_input):
    """Test 15: Rejects positive infinity."""
    corrupt = valid_numeric_input.copy()
    corrupt["age"] = float("inf")
    res = predict_user(corrupt)
    assert res["status"] == "error"
    assert "cannot be infinite" in res["message"]


# -------------------------------------------------------------------------
# Test 16: Negative infinity is rejected
# -------------------------------------------------------------------------
def test_16_negative_infinity_is_rejected(valid_numeric_input):
    """Test 16: Rejects negative infinity."""
    corrupt = valid_numeric_input.copy()
    corrupt["age"] = float("-inf")
    res = predict_user(corrupt)
    assert res["status"] == "error"
    assert "cannot be infinite" in res["message"]


# -------------------------------------------------------------------------
# Test 17: Prediction is either 0 or 1
# -------------------------------------------------------------------------
def test_17_prediction_is_binary(valid_numeric_input, valid_string_input):
    """Test 17: Prediction discrete label is strictly 0 or 1."""
    res1 = predict_user(valid_numeric_input)
    res2 = predict_user(valid_string_input)
    assert res1["prediction"] in [0, 1]
    assert res2["prediction"] in [0, 1]


# -------------------------------------------------------------------------
# Test 18: prediction_class is either NO or YES
# -------------------------------------------------------------------------
def test_18_prediction_class_is_no_or_yes(valid_numeric_input, valid_string_input):
    """Test 18: String class label is strictly 'NO' or 'YES'."""
    res1 = predict_user(valid_numeric_input)
    res2 = predict_user(valid_string_input)
    assert res1["prediction_class"] in ["NO", "YES"]
    assert res2["prediction_class"] in ["NO", "YES"]


# -------------------------------------------------------------------------
# Test 19: Probability is between 0 and 1
# -------------------------------------------------------------------------
def test_19_probability_range(valid_numeric_input, valid_string_input):
    """Test 19: Positive class probability is in [0.0, 1.0]."""
    res1 = predict_user(valid_numeric_input)
    res2 = predict_user(valid_string_input)
    assert 0.0 <= res1["probability"] <= 1.0
    assert 0.0 <= res2["probability"] <= 1.0


# -------------------------------------------------------------------------
# Test 20: Risk percentage is between 0 and 100
# -------------------------------------------------------------------------
def test_20_risk_percentage_range(valid_numeric_input, valid_string_input):
    """Test 20: Risk percentage is in [0.0, 100.0] and matches probability * 100."""
    res1 = predict_user(valid_numeric_input)
    assert 0.0 <= res1["risk_percentage"] <= 100.0
    assert np.isclose(res1["risk_percentage"], res1["probability"] * 100.0, atol=0.1)


# -------------------------------------------------------------------------
# Test 21: Prediction corresponds correctly to the 0.5 threshold
# -------------------------------------------------------------------------
def test_21_prediction_corresponds_to_threshold(valid_numeric_input, valid_string_input):
    """Test 21: Verifies prediction == 1 iff probability >= 0.50."""
    for data in [valid_numeric_input, valid_string_input]:
        res = predict_user(data)
        if res["probability"] >= DEFAULT_THRESHOLD:
            assert res["prediction"] == 1
            assert res["prediction_class"] == "YES"
            assert res["screening_interpretation"] == "Higher predicted risk"
        else:
            assert res["prediction"] == 0
            assert res["prediction_class"] == "NO"
            assert res["screening_interpretation"] == "Lower predicted risk"


# -------------------------------------------------------------------------
# Test 22: predict_user() and predict_clinical() return the same prediction
# -------------------------------------------------------------------------
def test_22_user_and_clinical_same_prediction(valid_numeric_input, valid_string_input):
    """Test 22: Identical prediction across user and clinical interfaces."""
    for data in [valid_numeric_input, valid_string_input]:
        res_u = predict_user(data)
        res_c = predict_clinical(data)
        assert res_u["prediction"] == res_c["prediction"]
        assert res_u["prediction_class"] == res_c["prediction_class"]


# -------------------------------------------------------------------------
# Test 23: predict_user() and predict_clinical() return the same probability
# -------------------------------------------------------------------------
def test_23_user_and_clinical_same_probability(valid_numeric_input, valid_string_input):
    """Test 23: Identical probabilities across user and clinical interfaces."""
    for data in [valid_numeric_input, valid_string_input]:
        res_u = predict_user(data)
        res_c = predict_clinical(data)
        assert res_u["probability"] == res_c["probability"]
        assert res_u["risk_percentage"] == res_c["risk_percentage"]


# -------------------------------------------------------------------------
# Test 24: Input feature order is deterministic
# -------------------------------------------------------------------------
def test_24_input_feature_order_is_deterministic(valid_numeric_input):
    """Test 24: Shuffled dictionary keys produce identical prediction output."""
    # Reverse dictionary key insertion order
    shuffled_keys = list(reversed(list(valid_numeric_input.keys())))
    shuffled_dict = {k: valid_numeric_input[k] for k in shuffled_keys}

    res_orig = predict_user(valid_numeric_input)
    res_shuffled = predict_user(shuffled_dict)

    assert res_orig == res_shuffled


# -------------------------------------------------------------------------
# Test 25: Reloading the model gives consistent predictions
# -------------------------------------------------------------------------
def test_25_reloading_model_consistency(valid_numeric_input):
    """Test 25: Forcing artifact reload produces identical numerical outputs."""
    res1 = predict_user(valid_numeric_input)
    # Force reload from disk
    load_artifacts(force_reload=True)
    res2 = predict_user(valid_numeric_input)

    assert res1 == res2


# -------------------------------------------------------------------------
# Test 26: Real dataset rows can pass through the prediction engine
# -------------------------------------------------------------------------
def test_26_real_dataset_rows_smoke_test():
    """Test 26: Real rows from survey_lung_cancer.csv execute successfully."""
    raw_df = load_raw_lung_cancer_data()

    # Test row 0 (YES ground truth)
    row_yes = {
        "gender": raw_df.loc[0, "GENDER"],
        "age": float(raw_df.loc[0, "AGE"]),
        "smoking": int(raw_df.loc[0, "SMOKING"]),
        "yellow_fingers": int(raw_df.loc[0, "YELLOW_FINGERS"]),
        "anxiety": int(raw_df.loc[0, "ANXIETY"]),
        "peer_pressure": int(raw_df.loc[0, "PEER_PRESSURE"]),
        "chronic_disease": int(raw_df.loc[0, "CHRONIC DISEASE"]),
        "fatigue": int(raw_df.loc[0, "FATIGUE "]),
        "allergy": int(raw_df.loc[0, "ALLERGY "]),
        "wheezing": int(raw_df.loc[0, "WHEEZING"]),
        "alcohol_consuming": int(raw_df.loc[0, "ALCOHOL CONSUMING"]),
        "coughing": int(raw_df.loc[0, "COUGHING"]),
        "shortness_of_breath": int(raw_df.loc[0, "SHORTNESS OF BREATH"]),
        "swallowing_difficulty": int(raw_df.loc[0, "SWALLOWING DIFFICULTY"]),
        "chest_pain": int(raw_df.loc[0, "CHEST PAIN"]),
    }
    res_yes = predict_user(row_yes)
    assert res_yes["status"] == "success"
    assert res_yes["prediction"] == 1
    assert res_yes["prediction_class"] == "YES"
    assert res_yes["probability"] > 0.50

    # Test a NO ground truth row (e.g., index of first NO case)
    no_indices = raw_df[raw_df["LUNG_CANCER"] == "NO"].index
    assert len(no_indices) > 0
    first_no_idx = no_indices[0]

    row_no = {
        "gender": raw_df.loc[first_no_idx, "GENDER"],
        "age": float(raw_df.loc[first_no_idx, "AGE"]),
        "smoking": int(raw_df.loc[first_no_idx, "SMOKING"]),
        "yellow_fingers": int(raw_df.loc[first_no_idx, "YELLOW_FINGERS"]),
        "anxiety": int(raw_df.loc[first_no_idx, "ANXIETY"]),
        "peer_pressure": int(raw_df.loc[first_no_idx, "PEER_PRESSURE"]),
        "chronic_disease": int(raw_df.loc[first_no_idx, "CHRONIC DISEASE"]),
        "fatigue": int(raw_df.loc[first_no_idx, "FATIGUE "]),
        "allergy": int(raw_df.loc[first_no_idx, "ALLERGY "]),
        "wheezing": int(raw_df.loc[first_no_idx, "WHEEZING"]),
        "alcohol_consuming": int(raw_df.loc[first_no_idx, "ALCOHOL CONSUMING"]),
        "coughing": int(raw_df.loc[first_no_idx, "COUGHING"]),
        "shortness_of_breath": int(raw_df.loc[first_no_idx, "SHORTNESS OF BREATH"]),
        "swallowing_difficulty": int(raw_df.loc[first_no_idx, "SWALLOWING DIFFICULTY"]),
        "chest_pain": int(raw_df.loc[first_no_idx, "CHEST PAIN"]),
    }
    res_no = predict_user(row_no)
    assert res_no["status"] == "success"
    assert res_no["prediction"] in [0, 1]
    assert 0.0 <= res_no["probability"] <= 1.0


# -------------------------------------------------------------------------
# Test 27: Explicit rejection of boolean values
# -------------------------------------------------------------------------
def test_27_boolean_values_explicitly_rejected(valid_numeric_input):
    """Test 27: Rejects Python booleans (True/False) per prompt specifications."""
    corrupt_bool_sym = valid_numeric_input.copy()
    corrupt_bool_sym["smoking"] = True
    res = predict_user(corrupt_bool_sym)
    assert res["status"] == "error"
    assert "Boolean values" in res["message"]

    corrupt_bool_gender = valid_numeric_input.copy()
    corrupt_bool_gender["gender"] = False
    res_g = predict_user(corrupt_bool_gender)
    assert res_g["status"] == "error"
    assert "Boolean values" in res_g["message"]


# -------------------------------------------------------------------------
# Test 28: raise_on_error raises LungCancerValidationError
# -------------------------------------------------------------------------
def test_28_raise_on_error_exception(valid_numeric_input):
    """Test 28: Verifies raise_on_error=True raises typed exception."""
    corrupt = valid_numeric_input.copy()
    corrupt["age"] = -10
    with pytest.raises(LungCancerValidationError, match="Age must be positive"):
        predict_user(corrupt, raise_on_error=True)
