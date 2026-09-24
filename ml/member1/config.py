"""
Shared Configuration for Member 1 ML Pipelines
(Heart Disease, Diabetes, Stroke)

Provides reproducible parameters, directory paths, and metadata constants.
"""

from pathlib import Path

# Base Paths
MEMBER1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MEMBER1_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# Raw Dataset Paths (Centralized, Read-Only)
HEART_RAW_DATA_PATH = DATASET_DIR / "heart.csv"
DIABETES_RAW_DATA_PATH = DATASET_DIR / "diabetes.csv"
STROKE_RAW_DATA_PATH = DATASET_DIR / "stroke.csv"

# Global Reproducibility Configuration
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
N_SPLITS_CV = 5

# Target Column Names in Raw Data
HEART_TARGET_COL = "Heart Disease"
DIABETES_TARGET_COL = "Outcome"
STROKE_TARGET_COL = "stroke"

# Module Specific Subdirectories
def get_disease_paths(disease: str) -> dict:
    """Returns standardized directory paths for a disease module."""
    base = MEMBER1_DIR / disease
    return {
        "base": base,
        "data": base / "data",
        "notebooks": base / "notebooks",
        "src": base / "src",
        "models": base / "models",
        "reports": base / "reports",
        "eda_reports": base / "reports" / "eda",
    }
