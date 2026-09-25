"""
Diabetes Data Cleaning & Zero-Value Standardization Module.

Responsibilities:
- Ingest raw DataFrame and perform deterministic, reproducible data cleaning.
- Standardize column names (strip leading/trailing whitespace).
- Audit and convert ONLY physiologically implausible zero measurements to NaN:
    - Glucose (0 mg/dl is incompatible with life)
    - BloodPressure (0 mm Hg indicates circulatory collapse)
    - SkinThickness (triceps skinfold cannot be 0 mm)
    - Insulin (2-hour serum insulin cannot be absolute 0 uU/ml in viable subjects)
    - BMI (weight/height^2 cannot be 0)
- CRITICAL: Pregnancies = 0 represents valid nulliparous status and is PRESERVED.
- Do NOT impute NaN values in this phase (imputation belongs in leak-free pipeline Phase 8).
- Verify target column 'Outcome' contains valid binary values (0, 1).
- Return a fresh cleaned DataFrame copy without mutating the input.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Columns where 0 represents missing/unrecorded biological measurement
BIOLOGICAL_ZERO_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Columns where 0 is valid and must NOT be converted
VALID_ZERO_COLS = [
    "Pregnancies",
    "Outcome"
]

# Deterministic Target Class Mapping
DIABETES_TARGET_MAPPING = {
    0: 0,
    1: 1
}


def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Diabetes dataset deterministically.
    Converts biologically invalid zero values to np.nan while preserving valid zeros.

    Args:
        df (pd.DataFrame): Raw DataFrame loaded from dataset/diabetes.csv.

    Returns:
        pd.DataFrame: Cleaned working DataFrame with explicit NaNs for missing biological measurements.
    """
    # 1. Create a deep copy to ensure no mutation of original input
    cleaned_df = df.copy(deep=True)

    # 2. Strip whitespace from column headers
    cleaned_df.columns = cleaned_df.columns.str.strip()

    # 3. Verify and handle duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_rows - len(cleaned_df)
    if duplicates_removed > 0:
        print(f"[Diabetes Cleaning] Removed {duplicates_removed} duplicate rows.")

    # 4. Convert biologically impossible zeros to NaN
    conversion_counts = {}
    for col in BIOLOGICAL_ZERO_COLS:
        if col in cleaned_df.columns:
            zeros_mask = (cleaned_df[col] == 0)
            count = int(zeros_mask.sum())
            conversion_counts[col] = count
            cleaned_df.loc[zeros_mask, col] = np.nan

    # 5. Assert valid zero preservation
    if "Pregnancies" in cleaned_df.columns:
        if (cleaned_df["Pregnancies"] == 0).sum() == 0:
            raise ValueError("[Diabetes Cleaning] ERROR: Valid zeros in Pregnancies were erroneously removed!")

    # 6. Target validation
    target_col = "Outcome"
    if target_col in cleaned_df.columns:
        cleaned_df[target_col] = cleaned_df[target_col].astype(int)
        if not set(cleaned_df[target_col].unique()).issubset({0, 1}):
            raise ValueError("[Diabetes Cleaning] Unexpected target classes encountered in 'Outcome'!")

    return cleaned_df


if __name__ == "__main__":
    from ml.member1.config import DIABETES_RAW_DATA_PATH, get_disease_paths

    paths = get_disease_paths("diabetes")
    raw_df = pd.read_csv(DIABETES_RAW_DATA_PATH)
    cleaned = clean_diabetes_data(raw_df)

    output_path = paths["data"] / "cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"[Diabetes Cleaning] Cleaned dataset saved to {output_path} (Shape: {cleaned.shape})")
    print(f"[Diabetes Cleaning] Missing values after zero conversion:\n{cleaned.isnull().sum()[cleaned.isnull().sum() > 0]}")
