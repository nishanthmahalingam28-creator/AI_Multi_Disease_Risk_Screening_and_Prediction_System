"""
Lung Cancer Preprocessing Pipeline Tests.

Validates:
- Test 1: Raw dataset loads correctly
- Test 2: Column normalization produces expected canonical names
- Test 3: Target mapping is NO -> 0, YES -> 1
- Test 4: Gender mapping is FEMALE -> 0, MALE -> 1
- Test 5: All 13 binary symptom features contain only 0 and 1
- Test 6: Final predictor feature count is exactly 15
- Test 7: Target column is strictly excluded from X
- Test 8: Scaler is fitted only to training data (no leakage)
- Test 9: Test transformation works using training-fitted preprocessor
- Test 10: No NaN or infinite values exist in processed matrices
- Test 11: Train/test split is reproducible with random_state=42
- Test 12: Duplicate feature groups do not cross train/test boundaries
- Test 13: Conflicting feature-label group does not occur in both train and test
- Test 14: Preprocessor artifact exists and can be loaded
- Test 15: Smoke test with 1 YES and 1 NO sample
- Test 16: Reload test verifying numerical identity before and after serialization
- Test 17: Metadata artifacts exist and conform to schemas
- Test 18-21: Robust validation errors on corrupted or out-of-domain inputs
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from member3b.lung_cancer.src.data_loader import (
    load_raw_lung_cancer_data,
    get_default_dataset_path,
    TARGET_COLUMN_RAW,
    EXPECTED_SHA256,
)
from member3b.lung_cancer.src.preprocessing import (
    clean_lung_cancer_data,
    normalize_columns,
    split_lung_cancer_data,
    build_preprocessor,
    fit_preprocessor,
    transform_data,
    load_preprocessor,
    get_default_models_dir,
    prepare_lung_cancer_data,
    TARGET_COLUMN,
    TARGET_MAPPING,
    GENDER_MAPPING,
    EXPECTED_FEATURES,
    EXPECTED_NUM_FEATURES,
    SCALED_FEATURES,
    BINARY_FEATURES,
    SYMPTOM_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


# -------------------------------------------------------------------------
# Test 1: Raw dataset loads correctly
# -------------------------------------------------------------------------
def test_1_raw_dataset_loads_correctly():
    """Test 1: Verifies raw dataset loads without error, has 309 rows and 16 cols."""
    df = load_raw_lung_cancer_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (309, 16)
    assert TARGET_COLUMN_RAW in df.columns
    assert set(df[TARGET_COLUMN_RAW].unique()) == {"YES", "NO"}


# -------------------------------------------------------------------------
# Test 2: Column normalization produces expected names
# -------------------------------------------------------------------------
def test_2_column_normalization_produces_expected_names():
    """Test 2: Verifies column normalization cleans trailing and internal whitespace."""
    raw_df = load_raw_lung_cancer_data()
    df_norm = normalize_columns(raw_df)

    expected_cols = [
        "gender", "age", "smoking", "yellow_fingers", "anxiety",
        "peer_pressure", "chronic_disease", "fatigue", "allergy",
        "wheezing", "alcohol_consuming", "coughing", "shortness_of_breath",
        "swallowing_difficulty", "chest_pain", "lung_cancer"
    ]
    assert list(df_norm.columns) == expected_cols
    assert len(df_norm.columns) == len(set(df_norm.columns)), "Duplicate column names detected!"
    # Ensure no leading/trailing spaces and no uppercase
    for col in df_norm.columns:
        assert col == col.strip()
        assert col == col.lower()
        assert " " not in col


# -------------------------------------------------------------------------
# Test 3: Target mapping is NO -> 0, YES -> 1
# -------------------------------------------------------------------------
def test_3_target_mapping():
    """Test 3: Verifies target encoding maps NO -> 0 and YES -> 1."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)

    assert y.name == "lung_cancer"
    assert set(y.unique()) == {0, 1}
    assert (y == 0).sum() == 39
    assert (y == 1).sum() == 270

    # Cross-reference with raw dataframe
    norm_df = normalize_columns(raw_df)
    for idx in [0, 1, 2, 10, 20]:
        expected_val = 1 if norm_df.loc[idx, "lung_cancer"] == "YES" else 0
        assert y.iloc[idx] == expected_val


# -------------------------------------------------------------------------
# Test 4: Gender mapping is FEMALE -> 0, MALE -> 1
# -------------------------------------------------------------------------
def test_4_gender_mapping():
    """Test 4: Verifies gender encoding maps FEMALE -> 0 and MALE -> 1."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)

    assert set(X["gender"].unique()) == {0, 1}
    assert (X["gender"] == 1).sum() == 162  # MALE
    assert (X["gender"] == 0).sum() == 147  # FEMALE

    norm_df = normalize_columns(raw_df)
    for idx in range(10):
        expected_gender = 1 if norm_df.loc[idx, "gender"] == "MALE" else 0
        assert X.loc[idx, "gender"] == expected_gender


# -------------------------------------------------------------------------
# Test 5: All 13 binary features contain only 0/1
# -------------------------------------------------------------------------
def test_5_all_13_binary_features_contain_only_0_and_1():
    """Test 5: Verifies that each of the 13 symptom/survey features contains strictly {0, 1}."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)

    assert len(SYMPTOM_FEATURES) == 13
    for col in SYMPTOM_FEATURES:
        assert col in X.columns
        unique_vals = set(X[col].unique())
        assert unique_vals.issubset({0, 1}), f"Column {col} contains invalid values: {unique_vals}"


# -------------------------------------------------------------------------
# Test 6: Final feature count is exactly 15
# -------------------------------------------------------------------------
def test_6_final_feature_count_is_15():
    """Test 6: Verifies predictor matrix has exactly 15 columns matching canonical list."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)

    assert X.shape[1] == EXPECTED_NUM_FEATURES
    assert list(X.columns) == EXPECTED_FEATURES


# -------------------------------------------------------------------------
# Test 7: Target is excluded from X
# -------------------------------------------------------------------------
def test_7_target_is_excluded_from_x():
    """Test 7: Verifies target 'lung_cancer' is not included in the predictor matrix X."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)

    assert TARGET_COLUMN not in X.columns
    assert "LUNG_CANCER" not in X.columns


# -------------------------------------------------------------------------
# Test 8: Scaler is fitted only to training data
# -------------------------------------------------------------------------
def test_8_scaler_is_fitted_only_to_training_data():
    """Test 8: Verifies StandardScaler is fitted solely on X_train, preventing leakage."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)
    X_train, X_test, y_train, y_test = split_lung_cancer_data(X, y, test_size=0.20, random_state=42)

    preprocessor = build_preprocessor()
    fitted_prep = fit_preprocessor(preprocessor, X_train)

    scaler = fitted_prep.named_transformers_["age"]
    fitted_mean = scaler.mean_[0]
    expected_train_mean = X_train["age"].mean()
    full_dataset_mean = X["age"].mean()

    # Must match train mean exactly
    assert np.isclose(fitted_mean, expected_train_mean, atol=1e-6)
    # Must NOT match full dataset mean (proves no leak from test set)
    assert not np.isclose(fitted_mean, full_dataset_mean, atol=1e-3)


# -------------------------------------------------------------------------
# Test 9: Test transformation works using training-fitted preprocessor
# -------------------------------------------------------------------------
def test_9_test_transformation_works_using_training_fitted_preprocessor():
    """Test 9: Verifies transforming X_test succeeds with training-fitted scaler."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)
    X_train, X_test, y_train, y_test = split_lung_cancer_data(X, y, test_size=0.20, random_state=42)

    preprocessor = build_preprocessor()
    fitted_prep = fit_preprocessor(preprocessor, X_train)

    X_test_proc = transform_data(fitted_prep, X_test)
    assert isinstance(X_test_proc, np.ndarray)
    assert X_test_proc.shape == (len(X_test), 15)

    # Verify manual formula: (age - train_mean) / train_std
    scaler = fitted_prep.named_transformers_["age"]
    expected_scaled_age = (X_test["age"].values - scaler.mean_[0]) / scaler.scale_[0]
    np.testing.assert_allclose(X_test_proc[:, 1], expected_scaled_age, atol=1e-6)


# -------------------------------------------------------------------------
# Test 10: No NaN/infinite values exist in processed training/test matrices
# -------------------------------------------------------------------------
def test_10_no_nan_or_infinite_values_in_processed_matrices():
    """Test 10: Verifies processed arrays contain zero NaN and zero infinite values."""
    res = prepare_lung_cancer_data(save_artifacts=False)
    X_train_proc = res["X_train_processed"]
    X_test_proc = res["X_test_processed"]

    assert not np.isnan(X_train_proc).any()
    assert not np.isinf(X_train_proc).any()
    assert not np.isnan(X_test_proc).any()
    assert not np.isinf(X_test_proc).any()


# -------------------------------------------------------------------------
# Test 11: Train/test split is reproducible with random_state=42
# -------------------------------------------------------------------------
def test_11_train_test_split_is_reproducible():
    """Test 11: Verifies running split with random_state=42 yields identical partitions."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)

    X_tr1, X_te1, y_tr1, y_te1 = split_lung_cancer_data(X, y, test_size=0.20, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = split_lung_cancer_data(X, y, test_size=0.20, random_state=42)

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_frame_equal(X_te1, X_te2)
    pd.testing.assert_series_equal(y_tr1, y_tr2)
    pd.testing.assert_series_equal(y_te1, y_te2)

    assert len(X_tr1) == 247
    assert len(X_te1) == 62


# -------------------------------------------------------------------------
# Test 12: Duplicate feature groups do not cross train/test boundaries
# -------------------------------------------------------------------------
def test_12_duplicate_feature_groups_do_not_cross_boundaries():
    """Test 12: Verifies zero duplicate feature groups cross the train/test split boundary."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)
    X_train, X_test, y_train, y_test = split_lung_cancer_data(X, y, test_size=0.20, random_state=42)

    # Form feature vector tuples
    train_tuples = set(tuple(row) for row in X_train.itertuples(index=False))
    test_tuples = set(tuple(row) for row in X_test.itertuples(index=False))

    overlap = train_tuples.intersection(test_tuples)
    assert len(overlap) == 0, f"Found {len(overlap)} duplicate feature vectors leaking across split!"


# -------------------------------------------------------------------------
# Test 13: The conflicting feature-label group does not occur in both train and test
# -------------------------------------------------------------------------
def test_13_conflicting_feature_vector_does_not_occur_in_both_partitions():
    """Test 13: Verifies rows 257 and 272 (identical features, opposite labels) are in one partition."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)
    X_train, X_test, y_train, y_test = split_lung_cancer_data(X, y, test_size=0.20, random_state=42)

    in_train_257 = 257 in X_train.index
    in_train_272 = 272 in X_train.index
    in_test_257 = 257 in X_test.index
    in_test_272 = 272 in X_test.index

    # Both must be strictly in train (or both strictly in test), NEVER split across
    assert (in_train_257 and in_train_272) or (in_test_257 and in_test_272)
    assert in_train_257 and in_train_272, "Expected both conflicting rows to reside in training set"
    assert not in_test_257 and not in_test_272, "Conflicting rows must not contaminate held-out test set"


# -------------------------------------------------------------------------
# Test 14: Preprocessor artifact exists and can be loaded
# -------------------------------------------------------------------------
def test_14_preprocessor_artifact_exists_and_can_be_loaded():
    """Test 14: Verifies saved lung_cancer_preprocessor.joblib exists and loads cleanly."""
    models_dir = get_default_models_dir()
    artifact_path = models_dir / "lung_cancer_preprocessor.joblib"
    assert artifact_path.is_file(), f"Preprocessor artifact missing at {artifact_path}"

    loaded_prep = load_preprocessor(artifact_path)
    assert hasattr(loaded_prep, "transform")
    assert "age" in loaded_prep.named_transformers_


# -------------------------------------------------------------------------
# Test 15: Smoke test with 1 YES and 1 NO sample (Section 21)
# -------------------------------------------------------------------------
def test_15_preprocessing_smoke_test_yes_no_samples():
    """Test 15: Selects 1 YES and 1 NO sample, transforms with preprocessor, checks output integrity."""
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)

    yes_sample = X[y == 1].iloc[[0]]
    no_sample = X[y == 0].iloc[[0]]

    preprocessor = load_preprocessor()

    yes_transformed = transform_data(preprocessor, yes_sample)
    no_transformed = transform_data(preprocessor, no_sample)

    assert yes_transformed.shape == (1, 15)
    assert no_transformed.shape == (1, 15)
    assert not np.isnan(yes_transformed).any()
    assert not np.isinf(yes_transformed).any()
    assert not np.isnan(no_transformed).any()
    assert not np.isinf(no_transformed).any()


# -------------------------------------------------------------------------
# Test 16: Reload test verifying numerical identity before and after serialization (Section 22)
# -------------------------------------------------------------------------
def test_16_preprocessor_serialization_reload_identity():
    """Test 16: Verifies output of loaded preprocessor is numerically identical to memory preprocessor."""
    res = prepare_lung_cancer_data(save_artifacts=True)
    memory_prep = res["preprocessor"]
    loaded_prep = load_preprocessor()

    X_test = res["X_test"]
    out_memory = transform_data(memory_prep, X_test)
    out_loaded = transform_data(loaded_prep, X_test)

    np.testing.assert_array_equal(out_memory, out_loaded)


# -------------------------------------------------------------------------
# Test 17: Preprocessing and Split Metadata Artifacts Exist and Valid
# -------------------------------------------------------------------------
def test_17_metadata_artifacts_exist_and_valid():
    """Test 17: Verifies preprocessing_metadata.json and split_metadata.json exist and match schema."""
    models_dir = get_default_models_dir()
    prep_meta_path = models_dir / "preprocessing_metadata.json"
    split_meta_path = models_dir / "split_metadata.json"

    assert prep_meta_path.is_file()
    assert split_meta_path.is_file()

    with open(prep_meta_path, "r", encoding="utf-8") as f:
        prep_meta = json.load(f)
    assert prep_meta["disease"] == "lung_cancer"
    assert prep_meta["feature_count"] == 15
    assert prep_meta["scaled_features"] == ["age"]
    assert len(prep_meta["binary_features"]) == 14
    assert prep_meta["train_rows"] == 247
    assert prep_meta["test_rows"] == 62

    with open(split_meta_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)
    assert split_meta["random_state"] == 42
    assert split_meta["test_size"] == 0.20
    assert len(split_meta["train_indices_raw_csv_0_indexed"]) == 247
    assert len(split_meta["test_indices_raw_csv_0_indexed"]) == 62
    assert split_meta["duplicate_group_handling"]["overlapping_groups"] == 0


# -------------------------------------------------------------------------
# Test 18-21: Robust Validation Errors on Corrupted or Out-of-Domain Inputs
# -------------------------------------------------------------------------
def test_18_validation_error_on_unexpected_target():
    """Test 18: Rejects unexpected target classes."""
    raw_df = load_raw_lung_cancer_data()
    corrupt_df = raw_df.copy()
    corrupt_df.loc[0, TARGET_COLUMN_RAW] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unexpected target categories"):
        clean_lung_cancer_data(corrupt_df)


def test_19_validation_error_on_unexpected_gender():
    """Test 19: Rejects unexpected gender categories."""
    raw_df = load_raw_lung_cancer_data()
    corrupt_df = raw_df.copy()
    corrupt_df.loc[0, "GENDER"] = "OTHER"
    with pytest.raises(ValueError, match="Unexpected gender values"):
        clean_lung_cancer_data(corrupt_df)


def test_20_validation_error_on_invalid_symptom_values():
    """Test 20: Rejects symptom values not in {0, 1}."""
    raw_df = load_raw_lung_cancer_data()
    corrupt_df = raw_df.copy()
    corrupt_df.loc[0, "SMOKING"] = 2
    with pytest.raises(ValueError, match="has invalid values"):
        clean_lung_cancer_data(corrupt_df)


def test_21_validation_error_on_nan_values():
    """Test 21: Rejects DataFrames containing NaN values."""
    raw_df = load_raw_lung_cancer_data()
    corrupt_df = raw_df.copy()
    corrupt_df.loc[0, "AGE"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        clean_lung_cancer_data(corrupt_df)
