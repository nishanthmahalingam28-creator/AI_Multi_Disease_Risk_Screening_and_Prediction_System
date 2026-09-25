"""
Breast Cancer Prediction Module - Data Loader.

Responsible for locating, loading, and validating the raw Breast Cancer dataset
from repository-relative paths without mutating the original CSV file.
"""

from pathlib import Path
from typing import Optional, Union, Set
import pandas as pd

# Expected target metadata
TARGET_COLUMN = "diagnosis"
EXPECTED_TARGET_VALUES: Set[str] = {"B", "M"}

# Standard 30 morphometric feature columns
EXPECTED_PREDICTOR_FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]


def get_default_dataset_path() -> Path:
    """
    Derives the repository root and returns the path to the raw breast cancer dataset.
    This guarantees path portability across diverse operating environments and clones.
    """
    # File is at: member3b/breast_cancer/src/data_loader.py
    # parents: 0=src, 1=breast_cancer, 2=member3b, 3=repo_root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "dataset" / "breast.csv"


def load_raw_breast_cancer_data(
    filepath: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Loads and performs initial structural integrity validation on the raw breast cancer dataset.

    Args:
        filepath: Optional path to the CSV file. If None, resolves to default repository path.

    Returns:
        pd.DataFrame: A copy of the loaded raw dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns or expected target classes are missing or invalid.
    """
    target_path = Path(filepath) if filepath is not None else get_default_dataset_path()

    if not target_path.exists():
        raise FileNotFoundError(
            f"Breast Cancer dataset not found at resolved location: {target_path}"
        )

    # Read CSV without mutating file
    df = pd.read_csv(target_path)

    # 1. Target column presence check
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Crucial target column '{TARGET_COLUMN}' is missing from the dataset. Found: {list(df.columns)}"
        )

    # 2. Target class validation
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    unexpected_targets = unique_targets - EXPECTED_TARGET_VALUES
    if unexpected_targets:
        raise ValueError(
            f"Unexpected target categories encountered in '{TARGET_COLUMN}': {unexpected_targets}. "
            f"Expected strictly a subset of: {EXPECTED_TARGET_VALUES}"
        )

    # 3. Missing target values check
    if df[TARGET_COLUMN].isnull().any():
        missing_count = int(df[TARGET_COLUMN].isnull().sum())
        raise ValueError(f"Found {missing_count} missing values in target column '{TARGET_COLUMN}'.")

    # 4. Check presence of all 30 expected clinical predictor columns
    missing_features = [col for col in EXPECTED_PREDICTOR_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Dataset is missing {len(missing_features)} expected predictor columns: {missing_features}"
        )

    return df.copy()
