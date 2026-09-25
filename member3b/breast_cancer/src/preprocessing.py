"""
Breast Cancer Prediction Module - Preprocessing Pipeline.

Implements a leak-safe, reproducible data cleaning, validation, splitting,
and scaling pipeline for the Wisconsin Diagnostic Breast Cancer dataset.
Strictly fits all scaling transformers on the training partition only.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from member3b.breast_cancer.src.data_loader import (
    load_raw_breast_cancer_data,
    TARGET_COLUMN,
    EXPECTED_PREDICTOR_FEATURES,
)

# Pipeline configuration constants
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
TARGET_MAPPING: Dict[str, int] = {"B": 0, "M": 1}
INVERSE_TARGET_MAPPING: Dict[int, str] = {0: "B", 1: "M"}
DROPPED_COLUMNS: List[str] = ["id", "Unnamed: 32"]
EXPECTED_NUM_FEATURES: int = 30


def get_default_models_dir() -> Path:
    """Returns the default directory for saving preprocessor artifacts."""
    repo_root = Path(__file__).resolve().parents[3]
    models_dir = repo_root / "member3b" / "breast_cancer" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def clean_breast_cancer_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans raw breast cancer data by removing non-predictive identifiers,
    empty parsing artifact columns, separating features from target,
    and deterministically encoding target classes.

    Args:
        df: Raw DataFrame containing breast cancer data.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Cleaned feature matrix X and encoded target vector y.

    Raises:
        ValueError: If target column is missing or contains unexpected classes.
    """
    df_clean = df.copy()

    # 1. Validate target column presence
    if TARGET_COLUMN not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    # 2. Extract and encode target deterministically
    raw_target = df_clean[TARGET_COLUMN]
    invalid_targets = set(raw_target.unique()) - set(TARGET_MAPPING.keys())
    if invalid_targets:
        raise ValueError(
            f"Unexpected target categories encountered in '{TARGET_COLUMN}': {invalid_targets}. "
            f"Expected strictly a subset of {list(TARGET_MAPPING.keys())}."
        )

    y = raw_target.map(TARGET_MAPPING).astype(int)
    y.name = TARGET_COLUMN

    # 3. Drop non-predictive columns: 'id' and 'Unnamed: 32'
    cols_to_drop = [TARGET_COLUMN]
    for col in DROPPED_COLUMNS:
        if col in df_clean.columns:
            cols_to_drop.append(col)

    X = df_clean.drop(columns=cols_to_drop)

    # 4. Enforce expected feature columns order and count
    validate_cleaned_data(X, y)

    return X, y


def validate_cleaned_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Validates that the cleaned predictor matrix and target vector meet medical-grade standards.

    Args:
        X: Predictor features DataFrame.
        y: Encoded target Series.

    Raises:
        ValueError: If any validation rule is violated.
    """
    # Verify no forbidden columns slipped in
    for col in DROPPED_COLUMNS:
        if col in X.columns:
            raise ValueError(f"Forbidden column '{col}' detected in predictor matrix.")

    if TARGET_COLUMN in X.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' accidentally included in predictors.")

    # Verify exact feature count
    if X.shape[1] != EXPECTED_NUM_FEATURES:
        raise ValueError(
            f"Expected exactly {EXPECTED_NUM_FEATURES} predictor features, but found {X.shape[1]}."
        )

    # Verify expected feature names match
    missing_expected = [c for c in EXPECTED_PREDICTOR_FEATURES if c not in X.columns]
    if missing_expected:
        raise ValueError(f"Cleaned feature matrix is missing expected features: {missing_expected}")

    # Verify no missing values in predictors
    if X.isnull().any().any():
        null_counts = X.isnull().sum()[X.isnull().sum() > 0].to_dict()
        raise ValueError(f"Cleaned feature matrix contains unexpected missing values: {null_counts}")

    # Verify all predictor features are numerical
    non_numeric_cols = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric_cols:
        raise ValueError(f"Non-numeric predictor columns found: {non_numeric_cols}")

    # Verify target has no missing values
    if y.isnull().any():
        raise ValueError("Target vector contains missing values.")

    # Verify target contains valid binary values
    unique_y = set(y.unique())
    if not unique_y.issubset({0, 1}):
        raise ValueError(f"Target vector contains invalid encoded values: {unique_y}")


def split_breast_cancer_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a leak-safe, reproducible stratified train/test split.

    Args:
        X: Predictor features.
        y: Encoded target vector.
        test_size: Proportion of dataset to include in test split (default: 0.20).
        random_state: Fixed random seed for complete reproducibility (default: 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Verify both classes exist in both partitions
    for name, partition_y in [("train", y_train), ("test", y_test)]:
        classes_present = set(partition_y.unique())
        if classes_present != {0, 1}:
            raise ValueError(
                f"Stratification error: {name} partition does not contain both classes. Found: {classes_present}"
            )

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> Pipeline:
    """
    Constructs an unfitted scikit-learn preprocessing pipeline.
    StandardScaler is selected to harmonize the widely differing feature scales
    (e.g., area vs. fractal dimension) without arbitrarily clipping medical outliers.

    Returns:
        Pipeline: Scikit-learn Pipeline with a StandardScaler step.
    """
    return Pipeline([
        ("scaler", StandardScaler())
    ])


def fit_preprocessor(preprocessor: Pipeline, X_train: pd.DataFrame) -> Pipeline:
    """
    Fits the preprocessing pipeline strictly on the training partition to eliminate data leakage.

    Args:
        preprocessor: Unfitted Pipeline.
        X_train: Training predictor features.

    Returns:
        Pipeline: Fitted preprocessing pipeline.
    """
    preprocessor.fit(X_train)
    return preprocessor


def transform_data(preprocessor: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Applies the fitted preprocessing transformation to feature data.
    Does not require target information.

    Args:
        preprocessor: Fitted Pipeline.
        X: Predictor features to transform.

    Returns:
        np.ndarray: Scaled feature array.
    """
    return preprocessor.transform(X)


def save_preprocessor(
    preprocessor: Pipeline,
    filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Serializes a genuinely fitted preprocessor to disk.

    Args:
        preprocessor: Fitted Pipeline.
        filepath: Optional target path. Defaults to member3b/breast_cancer/models/breast_cancer_preprocessor.joblib.

    Returns:
        Path: Path to the saved artifact.
    """
    out_path = Path(filepath) if filepath is not None else get_default_models_dir() / "breast_cancer_preprocessor.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_path)
    return out_path


def load_preprocessor(
    filepath: Optional[Union[str, Path]] = None
) -> Pipeline:
    """
    Loads a serialized preprocessor from disk.

    Args:
        filepath: Optional path to artifact.

    Returns:
        Pipeline: Loaded fitted preprocessing pipeline.
    """
    target_path = Path(filepath) if filepath is not None else get_default_models_dir() / "breast_cancer_preprocessor.joblib"
    if not target_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at: {target_path}")
    return joblib.load(target_path)


def save_preprocessing_metadata(
    metadata: Dict[str, Any],
    filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Persists actual preprocessing execution metadata to JSON.

    Args:
        metadata: Dictionary containing genuine preprocessing configuration and split statistics.
        filepath: Optional target path.

    Returns:
        Path: Path to saved metadata JSON.
    """
    out_path = Path(filepath) if filepath is not None else get_default_models_dir() / "preprocessing_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out_path


def prepare_breast_cancer_data(
    filepath: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    End-to-end data preparation workflow executing cleaning, validation,
    stratified splitting, and leak-safe scaling.

    Args:
        filepath: Optional path to raw dataset.
        save_artifacts: If True, serializes the fitted preprocessor and metadata.

    Returns:
        Dict[str, Any]: Container with raw splits, scaled arrays, fitted preprocessor, and metadata.
    """
    # 1. Load raw data
    raw_df = load_raw_breast_cancer_data(filepath=filepath)

    # 2. Clean data
    X, y = clean_breast_cancer_data(raw_df)

    # 3. Stratified Train/Test split
    X_train, X_test, y_train, y_test = split_breast_cancer_data(X, y)

    # 4. Leakage-safe scaling (fit ONLY on X_train)
    preprocessor = build_preprocessor()
    fitted_preprocessor = fit_preprocessor(preprocessor, X_train)

    X_train_scaled = transform_data(fitted_preprocessor, X_train)
    X_test_scaled = transform_data(fitted_preprocessor, X_test)

    # 5. Compile real metadata
    metadata: Dict[str, Any] = {
        "disease_name": "Breast Cancer Prediction",
        "dataset_path": str(filepath) if filepath is not None else "dataset/breast.csv",
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "inverse_target_mapping": {str(k): v for k, v in INVERSE_TARGET_MAPPING.items()},
        "dropped_columns": DROPPED_COLUMNS,
        "feature_names": list(X.columns),
        "feature_count": int(X.shape[1]),
        "scaler_type": "StandardScaler",
        "train_test_split_ratio": {
            "train": 1.0 - TEST_SIZE,
            "test": TEST_SIZE
        },
        "random_state": RANDOM_STATE,
        "total_samples": int(len(raw_df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_distribution_total": {
            "B (0)": int((y == 0).sum()),
            "M (1)": int((y == 1).sum())
        },
        "class_distribution_train": {
            "B (0)": int((y_train == 0).sum()),
            "M (1)": int((y_train == 1).sum())
        },
        "class_distribution_test": {
            "B (0)": int((y_test == 0).sum()),
            "M (1)": int((y_test == 1).sum())
        },
        "class_percentage_train": {
            "B (0)": round(float((y_train == 0).mean() * 100), 2),
            "M (1)": round(float((y_train == 1).mean() * 100), 2)
        },
        "class_percentage_test": {
            "B (0)": round(float((y_test == 0).mean() * 100), 2),
            "M (1)": round(float((y_test == 1).mean() * 100), 2)
        },
        "leakage_protection": "Scaler fitted exclusively on X_train; X_test transformed using fitted scaler.",
        "outlier_policy": "Preserved without truncation/clipping to maintain diagnostic signal of aggressive tumors.",
        "multicollinearity_policy": "All 30 continuous features preserved; algorithm-level regularization/tree-handling applied in Step 4.",
        "class_imbalance_policy": "Stratified splitting used; no SMOTE or artificial resampling applied."
    }

    # 6. Save artifacts if requested
    preprocessor_path = None
    metadata_path = None
    if save_artifacts:
        preprocessor_path = save_preprocessor(fitted_preprocessor)
        metadata_path = save_preprocessing_metadata(metadata)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "preprocessor": fitted_preprocessor,
        "metadata": metadata,
        "preprocessor_path": preprocessor_path,
        "metadata_path": metadata_path,
    }
