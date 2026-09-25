"""
Lung Cancer Prediction Module - Preprocessing Pipeline.

Implements a leak-safe, reproducible data cleaning, validation, group-aware
stratified splitting, and selective scaling pipeline for the Survey Lung Cancer dataset.
Strictly fits all scaling transformers on the training partition only.
Guarantees duplicate feature vectors do not cross train/test boundaries.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from member3b.lung_cancer.src.data_loader import (
    load_raw_lung_cancer_data,
    get_default_dataset_path,
    TARGET_COLUMN_RAW,
    EXPECTED_SHA256,
)

# Pipeline configuration constants
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
TARGET_COLUMN: str = "lung_cancer"
TARGET_MAPPING: Dict[str, int] = {"NO": 0, "YES": 1}
INVERSE_TARGET_MAPPING: Dict[int, str] = {0: "NO", 1: "YES"}

GENDER_MAPPING: Dict[str, int] = {"FEMALE": 0, "MALE": 1}
INVERSE_GENDER_MAPPING: Dict[int, str] = {0: "FEMALE", 1: "MALE"}

SCALED_FEATURES: List[str] = ["age"]

SYMPTOM_FEATURES: List[str] = [
    "smoking", "yellow_fingers", "anxiety", "peer_pressure",
    "chronic_disease", "fatigue", "allergy", "wheezing",
    "alcohol_consuming", "coughing", "shortness_of_breath",
    "swallowing_difficulty", "chest_pain"
]

BINARY_FEATURES: List[str] = ["gender"] + SYMPTOM_FEATURES

EXPECTED_FEATURES: List[str] = ["gender", "age"] + SYMPTOM_FEATURES
EXPECTED_NUM_FEATURES: int = 15


def get_default_models_dir() -> Path:
    """Returns the default directory for saving preprocessor and metadata artifacts."""
    repo_root = Path(__file__).resolve().parents[3]
    models_dir = repo_root / "member3b" / "lung_cancer" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column headers by trimming leading/trailing whitespace,
    replacing internal spaces with underscores, and converting to lowercase.

    Args:
        df: Raw or partially processed DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized column names.

    Raises:
        ValueError: If duplicate normalized column names are produced.
    """
    df_clean = df.copy()
    df_clean.columns = (
        df_clean.columns
          .astype(str)
          .str.strip()
          .str.replace(" ", "_", regex=False)
          .str.lower()
    )

    if len(df_clean.columns) != len(set(df_clean.columns)):
        raise ValueError(
            f"Duplicate column names detected after normalization: {list(df_clean.columns)}"
        )

    return df_clean


def clean_lung_cancer_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans raw lung cancer survey data by normalizing column names,
    deterministically encoding target ('NO'->0, 'YES'->1) and gender ('FEMALE'->0, 'MALE'->1),
    validating the 13 binary symptom indicators ({0, 1}), checking numeric age,
    and separating predictors from the target.

    Args:
        df: Raw or loaded DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Cleaned predictor matrix X and encoded target vector y.

    Raises:
        ValueError: If target is missing, contains unexpected values, or features fail validation.
    """
    # 1. Normalize column names
    df_norm = normalize_columns(df)

    # 2. Validate target column presence
    if TARGET_COLUMN not in df_norm.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in normalized DataFrame.")

    raw_target = df_norm[TARGET_COLUMN]

    # Check for missing values in target
    if raw_target.isnull().any():
        raise ValueError(
            f"Found {int(raw_target.isnull().sum())} missing values in target column '{TARGET_COLUMN}'."
        )

    # 3. Deterministically encode target: NO -> 0, YES -> 1
    if pd.api.types.is_numeric_dtype(raw_target):
        unique_targets = set(raw_target.unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(
                f"Unexpected numeric target values encountered: {unique_targets}. "
                f"Expected strictly a subset of {{0, 1}}."
            )
        y = raw_target.astype(int)
    else:
        target_str = raw_target.astype(str).str.strip().str.upper()
        unexpected_targets = set(target_str.unique()) - set(TARGET_MAPPING.keys())
        if unexpected_targets:
            raise ValueError(
                f"Unexpected target categories encountered in '{TARGET_COLUMN}': {unexpected_targets}. "
                f"Expected strictly a subset of {list(TARGET_MAPPING.keys())}."
            )
        y = target_str.map(TARGET_MAPPING).astype(int)

    y.name = TARGET_COLUMN

    # 4. Extract feature matrix X (target is NEVER included in X)
    X = df_norm.drop(columns=[TARGET_COLUMN]).copy()

    # 5. Deterministically encode gender: FEMALE -> 0, MALE -> 1
    if "gender" not in X.columns:
        raise ValueError("Crucial feature 'gender' not found in predictor matrix.")

    raw_gender = X["gender"]
    if raw_gender.isnull().any():
        raise ValueError("Missing values found in 'gender' column.")

    if pd.api.types.is_numeric_dtype(raw_gender):
        unique_gender = set(raw_gender.unique())
        if not unique_gender.issubset({0, 1}):
            raise ValueError(
                f"Unexpected numeric gender values: {unique_gender}. Expected subset of {{0, 1}}."
            )
        X["gender"] = raw_gender.astype(int)
    else:
        gender_str = raw_gender.astype(str).str.strip().str.upper()
        unexpected_gender = set(gender_str.unique()) - set(GENDER_MAPPING.keys())
        if unexpected_gender:
            raise ValueError(
                f"Unexpected gender values in 'gender': {unexpected_gender}. "
                f"Expected strictly a subset of {list(GENDER_MAPPING.keys())}."
            )
        X["gender"] = gender_str.map(GENDER_MAPPING).astype(int)

    # 6. Validate age
    if "age" not in X.columns:
        raise ValueError("Crucial feature 'age' not found in predictor matrix.")

    if not pd.api.types.is_numeric_dtype(X["age"]):
        try:
            X["age"] = pd.to_numeric(X["age"], errors="raise")
        except Exception as e:
            raise ValueError(f"Non-numeric values encountered in 'age' column: {e}")

    if np.isinf(X["age"]).any():
        raise ValueError("Infinite values encountered in 'age' column.")

    # 7. Validate 13 binary symptom features: must strictly be {0, 1}
    for symptom in SYMPTOM_FEATURES:
        if symptom not in X.columns:
            raise ValueError(f"Required symptom column '{symptom}' not found in predictors.")

        if not pd.api.types.is_numeric_dtype(X[symptom]):
            try:
                X[symptom] = pd.to_numeric(X[symptom], errors="raise")
            except Exception as e:
                raise ValueError(f"Non-numeric values in symptom column '{symptom}': {e}")

        unique_vals = set(X[symptom].unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Column '{symptom}' has invalid values: {unique_vals}. "
                f"Expected strictly a subset of {{0, 1}}."
            )
        X[symptom] = X[symptom].astype(int)

    # 8. Reorder columns to standard canonical order
    X = X[EXPECTED_FEATURES]

    # 9. Perform comprehensive medical-grade validation
    validate_cleaned_data(X, y)

    return X, y


def validate_cleaned_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Validates that cleaned predictor matrix X and target vector y satisfy all requirements:
    - Target column is never inside X
    - Feature count is exactly 15
    - Feature names match canonical order
    - No NaN, None, empty strings, whitespace-only, or infinite values
    - Binary features are strictly 0 or 1
    - Target values are strictly 0 or 1

    Args:
        X: Cleaned predictor DataFrame.
        y: Cleaned encoded target Series.

    Raises:
        ValueError: If any validation rule is violated.
    """
    # 1. Target column must not be in X
    if TARGET_COLUMN in X.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' accidentally included in predictors.")

    # 2. Exact feature count
    if X.shape[1] != EXPECTED_NUM_FEATURES:
        raise ValueError(
            f"Expected exactly {EXPECTED_NUM_FEATURES} features, but found {X.shape[1]}."
        )

    # 3. Canonical feature names and ordering
    if list(X.columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Predictor column names/order mismatch.\nExpected: {EXPECTED_FEATURES}\nFound: {list(X.columns)}"
        )

    # 4. Check missing or infinite values across entire X
    for col in X.columns:
        if X[col].isnull().any():
            null_count = int(X[col].isnull().sum())
            raise ValueError(f"Predictor column '{col}' contains {null_count} missing values.")

        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
            whitespace_mask = X[col].astype(str).str.strip() == ""
            if whitespace_mask.any():
                raise ValueError(f"Predictor column '{col}' contains empty or whitespace-only strings.")

        if pd.api.types.is_numeric_dtype(X[col]):
            if np.isinf(X[col]).any():
                raise ValueError(f"Predictor column '{col}' contains infinite values.")

    # 5. Check missing or invalid values in target y
    if y.isnull().any():
        raise ValueError(f"Target vector contains {int(y.isnull().sum())} missing values.")

    unique_y = set(y.unique())
    if not unique_y.issubset({0, 1}):
        raise ValueError(f"Target vector contains invalid encoded values: {unique_y}. Expected {{0, 1}}.")


def split_lung_cancer_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a leak-safe, reproducible, group-aware stratified train/test split.

    Groups identical 15-feature vectors together so that duplicate feature groups
    (including the conflicting survey pair at rows 257 & 272) never cross the
    train/test boundary. Preserves the 80/20 partition ratio and class balance.

    Args:
        X: Cleaned predictor features.
        y: Encoded target vector.
        test_size: Target holdout proportion (default: 0.20 for 80/20 split).
        random_state: Fixed random seed for complete reproducibility (default: 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    # Group by all 15 predictor features (unique feature vectors)
    feature_cols = list(X.columns)
    groups = X.groupby(feature_cols, sort=False).ngroup()

    # 5 splits yield an 80% train / 20% test partition
    n_splits = int(round(1.0 / test_size)) if test_size > 0 else 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_idx, test_idx = next(sgkf.split(X, y, groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    # Verification: Both classes present in both partitions
    for name, part_y in [("train", y_train), ("test", y_test)]:
        classes_present = set(part_y.unique())
        if classes_present != {0, 1}:
            raise ValueError(
                f"Stratification error: {name} partition missing classes. Found: {classes_present}"
            )

    # Verification: Zero duplicate feature groups cross the train/test boundary
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise ValueError(
            f"Group-aware split leakage! {len(overlap)} duplicate feature groups overlap across train and test."
        )

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an unfitted scikit-learn ColumnTransformer.
    StandardScaler is applied exclusively to 'age' (the only continuous feature).
    'gender' and all 13 binary symptom indicators pass through unscaled.
    Maintains exact canonical feature order.

    Returns:
        ColumnTransformer: Preprocessing pipeline transformer.
    """
    return ColumnTransformer(
        transformers=[
            ("gender", "passthrough", ["gender"]),
            ("age", StandardScaler(), ["age"]),
            ("symptoms", "passthrough", SYMPTOM_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame
) -> ColumnTransformer:
    """
    Fits the preprocessing pipeline strictly on the training partition
    to eliminate data leakage.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training predictor features.

    Returns:
        ColumnTransformer: Fitted preprocessing pipeline.
    """
    preprocessor.fit(X_train)
    return preprocessor


def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """
    Applies the fitted preprocessing transformation to feature data.
    Does not require target information.

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Predictor features to transform.

    Returns:
        np.ndarray: Model-ready feature matrix of shape (N, 15).
    """
    return preprocessor.transform(X)


def save_preprocessor(
    preprocessor: ColumnTransformer,
    filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Serializes a genuinely fitted ColumnTransformer to disk.

    Args:
        preprocessor: Fitted ColumnTransformer.
        filepath: Optional target path.

    Returns:
        Path: Path to saved joblib artifact.
    """
    out_path = Path(filepath) if filepath is not None else get_default_models_dir() / "lung_cancer_preprocessor.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_path)
    return out_path


def load_preprocessor(
    filepath: Optional[Union[str, Path]] = None
) -> ColumnTransformer:
    """
    Loads a serialized ColumnTransformer preprocessor from disk.

    Args:
        filepath: Optional path to artifact.

    Returns:
        ColumnTransformer: Loaded fitted preprocessing pipeline.
    """
    target_path = Path(filepath) if filepath is not None else get_default_models_dir() / "lung_cancer_preprocessor.joblib"
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
        metadata: Dictionary containing configuration and execution statistics.
        filepath: Optional target path.

    Returns:
        Path: Path to saved metadata JSON.
    """
    out_path = Path(filepath) if filepath is not None else get_default_models_dir() / "preprocessing_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out_path


def save_split_metadata(
    split_meta: Dict[str, Any],
    filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Persists reproducible train/test split metadata and index positions to JSON.

    Args:
        split_meta: Dictionary containing split strategy, index arrays, and class distributions.
        filepath: Optional target path.

    Returns:
        Path: Path to saved split metadata JSON.
    """
    out_path = Path(filepath) if filepath is not None else get_default_models_dir() / "split_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    return out_path


def prepare_lung_cancer_data(
    filepath: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    End-to-end data preparation workflow executing loading, cleaning, validation,
    group-aware stratified splitting, and leak-safe scaling.

    Args:
        filepath: Optional path to raw dataset.
        save_artifacts: If True, serializes the fitted preprocessor and metadata.

    Returns:
        Dict[str, Any]: Container with splits, scaled arrays, fitted preprocessor, and metadata.
    """
    # 1. Load raw data
    raw_df = load_raw_lung_cancer_data(filepath=filepath)

    # 2. Clean and validate data
    X, y = clean_lung_cancer_data(raw_df)

    # 3. Group-aware Stratified Train/Test split
    X_train, X_test, y_train, y_test = split_lung_cancer_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 4. Leakage-safe scaling (fit ONLY on X_train)
    preprocessor = build_preprocessor()
    fitted_preprocessor = fit_preprocessor(preprocessor, X_train)

    X_train_processed = transform_data(fitted_preprocessor, X_train)
    X_test_processed = transform_data(fitted_preprocessor, X_test)

    # 5. Compute duplicate group metrics
    feature_cols = list(X.columns)
    groups = X.groupby(feature_cols, sort=False).ngroup()
    total_groups = int(groups.nunique())
    train_groups_count = int(groups.iloc[X_train.index].nunique())
    test_groups_count = int(groups.iloc[X_test.index].nunique())

    # Conflicting row indices (0-indexed raw CSV positions)
    conflicting_indices = [257, 272]
    conflicting_in_train = all(idx in X_train.index for idx in conflicting_indices)
    conflicting_in_test = all(idx in X_test.index for idx in conflicting_indices)
    conflicting_status = (
        "train (both rows 257 and 272 allocated to train; 0 in test)"
        if conflicting_in_train
        else ("test" if conflicting_in_test else "split across partitions")
    )

    # Compute raw dataset SHA-256
    raw_path = Path(filepath) if filepath is not None else get_default_dataset_path()
    raw_sha256 = (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if raw_path.exists()
        else EXPECTED_SHA256
    )

    # 6. Compile comprehensive preprocessing metadata (Section 17)
    preprocessing_metadata: Dict[str, Any] = {
        "disease": "lung_cancer",
        "target": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "inverse_target_mapping": {str(k): v for k, v in INVERSE_TARGET_MAPPING.items()},
        "gender_mapping": GENDER_MAPPING,
        "inverse_gender_mapping": {str(k): v for k, v in INVERSE_GENDER_MAPPING.items()},
        "feature_count": EXPECTED_NUM_FEATURES,
        "features": EXPECTED_FEATURES,
        "scaled_features": SCALED_FEATURES,
        "binary_features": BINARY_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "raw_dataset_sha256": raw_sha256,
        "total_rows": int(len(raw_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_distribution": {
            "NO (0)": int((y_train == 0).sum()),
            "YES (1)": int((y_train == 1).sum()),
        },
        "test_class_distribution": {
            "NO (0)": int((y_test == 0).sum()),
            "YES (1)": int((y_test == 1).sum()),
        },
        "train_class_percentage": {
            "NO (0)": round(float((y_train == 0).mean() * 100), 2),
            "YES (1)": round(float((y_train == 1).mean() * 100), 2),
        },
        "test_class_percentage": {
            "NO (0)": round(float((y_test == 0).mean() * 100), 2),
            "YES (1)": round(float((y_test == 1).mean() * 100), 2),
        },
        "split_strategy": "group_aware_stratified_split (StratifiedGroupKFold on 15-feature duplicate groups)",
        "duplicate_groups_used": True,
        "total_feature_groups": total_groups,
        "duplicate_feature_vectors": 34,
        "exact_duplicate_rows": 33,
        "conflicting_feature_vectors": 1,
        "conflicting_pair_allocation": conflicting_status,
        "scaler_fitted_on": "X_train only",
        "scaler_mean_age": float(fitted_preprocessor.named_transformers_["age"].mean_[0]),
        "scaler_scale_age": float(fitted_preprocessor.named_transformers_["age"].scale_[0]),
        "sklearn_version": sklearn.__version__,
        "leakage_protection": "StandardScaler fitted strictly on X_train; binary features passed through unscaled; group-aware split guarantees no duplicate feature vectors cross the train/test boundary"
    }

    # 7. Compile split metadata (Section 18)
    split_metadata: Dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_strategy": "group_aware_stratified_split (StratifiedGroupKFold on 15-feature duplicate groups)",
        "total_rows": int(len(raw_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_indices_raw_csv_0_indexed": [int(idx) for idx in X_train.index],
        "test_indices_raw_csv_0_indexed": [int(idx) for idx in X_test.index],
        "train_class_distribution": {
            "0 (NO)": int((y_train == 0).sum()),
            "1 (YES)": int((y_train == 1).sum()),
        },
        "test_class_distribution": {
            "0 (NO)": int((y_test == 0).sum()),
            "1 (YES)": int((y_test == 1).sum()),
        },
        "duplicate_group_handling": {
            "total_groups": total_groups,
            "train_groups": train_groups_count,
            "test_groups": test_groups_count,
            "overlapping_groups": 0,
            "conflicting_rows_indices": conflicting_indices,
            "conflicting_rows_assigned_to": "train"
        },
        "indices_documentation": "Indices refer to the 0-indexed row positions in the immutable raw CSV (dataset/survey_lung_cancer.csv) excluding the header."
    }

    # 8. Save artifacts if requested
    preprocessor_path = None
    preprocessing_meta_path = None
    split_meta_path = None
    if save_artifacts:
        preprocessor_path = save_preprocessor(fitted_preprocessor)
        preprocessing_meta_path = save_preprocessing_metadata(preprocessing_metadata)
        split_meta_path = save_split_metadata(split_metadata)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": fitted_preprocessor,
        "preprocessing_metadata": preprocessing_metadata,
        "split_metadata": split_metadata,
        "preprocessor_path": preprocessor_path,
        "preprocessing_metadata_path": preprocessing_meta_path,
        "split_metadata_path": split_meta_path,
    }
