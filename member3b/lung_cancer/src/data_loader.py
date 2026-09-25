"""
Lung Cancer Prediction Module - Data Loader.

Responsible for locating, loading, and validating the raw Lung Cancer dataset
from repository-relative paths without mutating the original CSV file.
"""

from pathlib import Path
from typing import Optional, Union, Set
import pandas as pd

# Expected raw metadata
EXPECTED_SHA256: str = "b5df44c7a33d095457bd67d59806ce96797d2aef591781af6ef4f1d93c5ac3d2"
TARGET_COLUMN_RAW: str = "LUNG_CANCER"
EXPECTED_TARGET_VALUES: Set[str] = {"YES", "NO"}

# Standard 16 raw column headers
EXPECTED_RAW_COLUMNS = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ", "WHEEZING",
    "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY", "CHEST PAIN", "LUNG_CANCER"
]


def get_default_dataset_path() -> Path:
    """
    Derives the repository root and returns the path to the raw lung cancer dataset.
    This guarantees path portability across diverse operating environments and clones.
    """
    # File is at: member3b/lung_cancer/src/data_loader.py
    # parents: 0=src, 1=lung_cancer, 2=member3b, 3=repo_root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "dataset" / "survey_lung_cancer.csv"


def load_raw_lung_cancer_data(
    filepath: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Loads and performs initial structural integrity validation on the raw lung cancer dataset.

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
            f"Lung Cancer dataset not found at resolved location: {target_path}"
        )

    # Read CSV without mutating original file
    df = pd.read_csv(target_path)

    # 1. Target column presence check (check both raw and normalized variations)
    target_found = None
    for candidate in [TARGET_COLUMN_RAW, "lung_cancer", "LUNG_CANCER"]:
        if candidate in df.columns:
            target_found = candidate
            break

    if target_found is None:
        raise ValueError(
            f"Crucial target column '{TARGET_COLUMN_RAW}' is missing from the dataset. Found: {list(df.columns)}"
        )

    # 2. Target class validation
    unique_targets = set(df[target_found].dropna().unique())
    unexpected_targets = unique_targets - EXPECTED_TARGET_VALUES
    if unexpected_targets:
        raise ValueError(
            f"Unexpected target categories encountered in '{target_found}': {unexpected_targets}. "
            f"Expected strictly a subset of: {EXPECTED_TARGET_VALUES}"
        )

    # 3. Missing target values check
    if df[target_found].isnull().any():
        missing_count = int(df[target_found].isnull().sum())
        raise ValueError(f"Found {missing_count} missing values in target column '{target_found}'.")

    return df.copy()
