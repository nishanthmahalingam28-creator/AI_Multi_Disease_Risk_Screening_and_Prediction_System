"""
Unit tests for the Breast Cancer Data Loader and Preprocessing Pipeline.

Validates data integrity, leak-free preprocessing, deterministic encoding,
stratified train/test partitioning, and raw dataset immutability.
"""

import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from member3b.breast_cancer.src.data_loader import (
    load_raw_breast_cancer_data,
    get_default_dataset_path,
    TARGET_COLUMN,
    EXPECTED_PREDICTOR_FEATURES,
)
from member3b.breast_cancer.src.preprocessing import (
    clean_breast_cancer_data,
    validate_cleaned_data,
    split_breast_cancer_data,
    build_preprocessor,
    fit_preprocessor,
    transform_data,
    save_preprocessor,
    load_preprocessor,
    prepare_breast_cancer_data,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_MAPPING,
    DROPPED_COLUMNS,
    EXPECTED_NUM_FEATURES,
)

# Reference SHA-256 hash for raw dataset/breast.csv
EXPECTED_RAW_DATASET_SHA256 = "1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33"


def test_raw_dataset_exists_and_unmodified():
    """Verify that dataset/breast.csv exists and remains bit-for-bit unmodified."""
    raw_path = get_default_dataset_path()
    assert raw_path.exists(), f"Raw dataset not found at {raw_path}"

    with open(raw_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    assert actual_hash == EXPECTED_RAW_DATASET_SHA256, (
        f"Raw dataset integrity violated! Expected hash {EXPECTED_RAW_DATASET_SHA256}, "
        f"got {actual_hash}"
    )


def test_load_raw_breast_cancer_data():
    """Verify raw data loading and basic shape validation."""
    df = load_raw_breast_cancer_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 569
    assert df.shape[1] == 33
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()) == {"B", "M"}


def test_data_cleaning_and_column_removal():
    """Verify id and Unnamed: 32 removal, and exact 30 predictor count."""
    df = load_raw_breast_cancer_data()
    X, y = clean_breast_cancer_data(df)

    # Check dropped columns are absent
    for dropped_col in DROPPED_COLUMNS:
        assert dropped_col not in X.columns, f"Forbidden column '{dropped_col}' found in X"

    # Check target column is absent from X
    assert TARGET_COLUMN not in X.columns, "Target column found in X"

    # Check feature count and identity
    assert X.shape[1] == EXPECTED_NUM_FEATURES
    assert list(X.columns) == EXPECTED_PREDICTOR_FEATURES

    # Check no missing values
    assert not X.isnull().any().any(), "Missing values detected in X"


def test_target_encoding():
    """Verify deterministic target mapping (B -> 0, M -> 1)."""
    df = load_raw_breast_cancer_data()
    _, y = clean_breast_cancer_data(df)

    assert set(y.unique()) == {0, 1}
    assert (y == 0).sum() == 357  # Benign count
    assert (y == 1).sum() == 212  # Malignant count


def test_target_validation_errors():
    """Verify that unexpected target categories raise an informative error."""
    df = load_raw_breast_cancer_data()
    df_corrupted = df.copy()
    df_corrupted.loc[0, TARGET_COLUMN] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unexpected target categories"):
        clean_breast_cancer_data(df_corrupted)


def test_train_test_split_reproducibility_and_stratification():
    """Verify train/test split is reproducible, stratified, and preserves class distribution."""
    df = load_raw_breast_cancer_data()
    X, y = clean_breast_cancer_data(df)

    X_train_1, X_test_1, y_train_1, y_test_1 = split_breast_cancer_data(X, y)
    X_train_2, X_test_2, y_train_2, y_test_2 = split_breast_cancer_data(X, y)

    # Check reproducibility
    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)

    # Check sample counts (80% / 20%)
    assert len(X_train_1) == 455
    assert len(X_test_1) == 114
    assert len(X_train_1) + len(X_test_1) == 569

    # Check stratification: both classes present and ratios match within 1%
    train_m_ratio = (y_train_1 == 1).mean()
    test_m_ratio = (y_test_1 == 1).mean()
    total_m_ratio = (y == 1).mean()

    assert abs(train_m_ratio - total_m_ratio) < 0.01
    assert abs(test_m_ratio - total_m_ratio) < 0.01
    assert set(y_train_1.unique()) == {0, 1}
    assert set(y_test_1.unique()) == {0, 1}


def test_preprocessor_fitting_and_transformation_no_leakage():
    """
    Verify that the preprocessor is fitted strictly on X_train,
    successfully scales X_test without seeing y, and standardizes properly.
    """
    df = load_raw_breast_cancer_data()
    X, y = clean_breast_cancer_data(df)
    X_train, X_test, y_train, y_test = split_breast_cancer_data(X, y)

    preprocessor = build_preprocessor()
    fitted_preprocessor = fit_preprocessor(preprocessor, X_train)

    # Transform without target
    X_train_scaled = transform_data(fitted_preprocessor, X_train)
    X_test_scaled = transform_data(fitted_preprocessor, X_test)

    # Check transformed shapes
    assert X_train_scaled.shape == (455, 30)
    assert X_test_scaled.shape == (114, 30)

    # Check training data is standardized (mean approx 0, std approx 1)
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), np.zeros(30), atol=1e-7)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), np.ones(30), atol=1e-7)

    # Check test data mean is close but NOT strictly 0 (proves test was NOT fitted)
    test_means = X_test_scaled.mean(axis=0)
    assert not np.allclose(test_means, np.zeros(30), atol=1e-7), (
        "Test data means are identically zero; indicates data leakage (scaler fitted on test)!"
    )


def test_preprocessor_serialization_and_reloading(tmp_path):
    """Verify that a fitted preprocessor can be saved and loaded with identical transforms."""
    df = load_raw_breast_cancer_data()
    X, y = clean_breast_cancer_data(df)
    X_train, X_test, _, _ = split_breast_cancer_data(X, y)

    preprocessor = fit_preprocessor(build_preprocessor(), X_train)
    artifact_path = tmp_path / "test_preprocessor.joblib"

    save_preprocessor(preprocessor, filepath=artifact_path)
    assert artifact_path.exists()

    loaded_preprocessor = load_preprocessor(filepath=artifact_path)
    original_transform = transform_data(preprocessor, X_test)
    loaded_transform = transform_data(loaded_preprocessor, X_test)

    np.testing.assert_array_equal(original_transform, loaded_transform)


def test_prepare_breast_cancer_data_end_to_end():
    """Verify the full end-to-end data preparation workflow."""
    results = prepare_breast_cancer_data(save_artifacts=True)

    assert "X_train" in results
    assert "X_test" in results
    assert "y_train" in results
    assert "y_test" in results
    assert "X_train_scaled" in results
    assert "X_test_scaled" in results
    assert "preprocessor" in results
    assert "metadata" in results

    # Verify metadata fields
    meta = results["metadata"]
    assert meta["disease_name"] == "Breast Cancer Prediction"
    assert meta["feature_count"] == 30
    assert meta["train_samples"] == 455
    assert meta["test_samples"] == 114
    assert meta["class_distribution_total"]["B (0)"] == 357
    assert meta["class_distribution_total"]["M (1)"] == 212
