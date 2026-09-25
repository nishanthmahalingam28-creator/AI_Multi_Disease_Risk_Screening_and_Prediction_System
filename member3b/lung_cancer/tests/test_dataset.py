"""
Dataset analysis and data-quality verification tests for Lung Cancer module.

Validates:
- Raw dataset existence and SHA-256 integrity
- Exact column names, lengths, and trailing whitespace anomalies
- Target distribution and class balance metrics
- Data types, absence of missing values, and value domains
- Presence and validity of EDA generated reports
"""

from pathlib import Path
import hashlib
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "dataset" / "survey_lung_cancer.csv"
REPORTS_DIR = REPO_ROOT / "member3b" / "lung_cancer" / "reports"

EXPECTED_SHA256 = "181ccdd5e12900a9a428e1ad7a570227617d41266c4dc7930465ebfa009a3efd"
EXPECTED_ROWS = 309
EXPECTED_COLS = 16

EXPECTED_COLUMNS = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ", "WHEEZING",
    "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY", "CHEST PAIN", "LUNG_CANCER"
]


def test_raw_lung_cancer_dataset_exists_and_unmodified():
    """Verifies raw dataset exists and matches frozen SHA-256 hash."""
    assert DATASET_PATH.is_file(), f"Dataset file missing at {DATASET_PATH}"
    raw_bytes = DATASET_PATH.read_bytes()
    computed_hash = hashlib.sha256(raw_bytes).hexdigest()
    assert computed_hash == EXPECTED_SHA256, (
        f"Dataset hash mismatch! Expected {EXPECTED_SHA256}, got {computed_hash}"
    )
    assert len(raw_bytes) == 12188


def test_lung_cancer_dataset_shape_and_columns():
    """Verifies dataset dimensions and identifies exact column header strings."""
    df = pd.read_csv(DATASET_PATH)
    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLS)
    assert list(df.columns) == EXPECTED_COLUMNS

    # Explicit check for known header whitespace anomalies
    assert "FATIGUE " in df.columns
    assert "FATIGUE" not in df.columns
    assert df.columns[7] == "FATIGUE "
    assert df.columns[7].endswith(" ")

    assert "ALLERGY " in df.columns
    assert "ALLERGY" not in df.columns
    assert df.columns[8] == "ALLERGY "
    assert df.columns[8].endswith(" ")


def test_lung_cancer_target_values_and_distribution():
    """Verifies target column identity, allowed labels, and exact class frequencies."""
    df = pd.read_csv(DATASET_PATH)
    assert "LUNG_CANCER" in df.columns
    target = df["LUNG_CANCER"]

    assert set(target.unique()) == {"YES", "NO"}
    counts = target.value_counts().to_dict()
    assert counts["YES"] == 270
    assert counts["NO"] == 39

    # Imbalance ratio ~6.92
    ratio = counts["YES"] / counts["NO"]
    assert round(ratio, 2) == 6.92


def test_lung_cancer_feature_types_and_missing_values():
    """Verifies zero missing values, demographic bounds, and binary symptom domains."""
    df = pd.read_csv(DATASET_PATH)

    # 1. Zero missing values across all columns
    assert df.isnull().sum().sum() == 0

    # 2. Gender domain
    assert set(df["GENDER"].unique()) == {"MALE", "FEMALE"}
    assert df["GENDER"].value_counts()["MALE"] == 162
    assert df["GENDER"].value_counts()["FEMALE"] == 147

    # 3. Age domain
    assert df["AGE"].min() == 21
    assert df["AGE"].max() == 87
    assert 62.0 <= df["AGE"].mean() <= 63.5

    # 4. Binary symptom/behavior features (all encoded as 0 or 1 in this dataset)
    binary_cols = [c for c in df.columns if c not in ["GENDER", "AGE", "LUNG_CANCER"]]
    assert len(binary_cols) == 13
    for col in binary_cols:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({0, 1}), f"Column {col} has unexpected values: {unique_vals}"


def test_lung_cancer_eda_reports_exist():
    """Verifies that all required EDA visualization reports have been generated and are non-empty."""
    expected_reports = [
        "target_distribution.png",
        "feature_distribution.png",
        "feature_target_relationships.png",
    ]
    for filename in expected_reports:
        report_file = REPORTS_DIR / filename
        assert report_file.is_file(), f"Report {filename} missing in {REPORTS_DIR}"
        assert report_file.stat().st_size > 10000, f"Report {filename} appears incomplete or empty"
