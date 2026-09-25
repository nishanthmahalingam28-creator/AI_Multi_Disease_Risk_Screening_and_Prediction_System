"""
Heart Disease Data Cleaning Module.

Responsibilities:
- Ingest raw DataFrame and perform deterministic, reproducible data cleaning.
- Standardize column names (strip leading/trailing whitespace).
- Verify duplicate rows (no duplicates present; drop if found).
- Map target column 'Heart Disease' deterministically: 'Absence' -> 0, 'Presence' -> 1.
- Preserve discrete clinical scales without artificial mutation.
- Return a fresh cleaned DataFrame copy without altering the input.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Deterministic Target Class Mapping
HEART_TARGET_MAPPING = {
    "Absence": 0,
    "Presence": 1
}

HEART_REVERSE_MAPPING = {
    0: "Absence",
    1: "Presence"
}


def clean_heart_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Heart Disease dataset deterministically.

    Args:
        df (pd.DataFrame): Raw DataFrame loaded from dataset/heart.csv.

    Returns:
        pd.DataFrame: Cleaned working DataFrame ready for EDA and leak-free preprocessing.
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
        print(f"[Heart Cleaning] Removed {duplicates_removed} duplicate rows.")

    # 4. Deterministic Target Encoding
    target_col = "Heart Disease"
    if target_col in cleaned_df.columns:
        if cleaned_df[target_col].dtype == object or isinstance(cleaned_df[target_col].iloc[0], str):
            cleaned_df[target_col] = cleaned_df[target_col].map(HEART_TARGET_MAPPING)
            if cleaned_df[target_col].isnull().any():
                raise ValueError("[Heart Cleaning] Unmapped values encountered in target column 'Heart Disease'!")
            cleaned_df[target_col] = cleaned_df[target_col].astype(int)

    # 5. Type assertions for clinical features
    int_cols = [
        "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
        "FBS over 120", "EKG results", "Max HR", "Exercise angina",
        "Slope of ST", "Number of vessels fluro", "Thallium"
    ]
    for col in int_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(int)

    if "ST depression" in cleaned_df.columns:
        cleaned_df["ST depression"] = cleaned_df["ST depression"].astype(float)

    return cleaned_df


if __name__ == "__main__":
    from ml.member1.config import HEART_RAW_DATA_PATH, get_disease_paths

    paths = get_disease_paths("heart")
    raw_df = pd.read_csv(HEART_RAW_DATA_PATH)
    cleaned = clean_heart_data(raw_df)

    output_path = paths["data"] / "cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"[Heart Cleaning] Cleaned dataset saved to {output_path} (Shape: {cleaned.shape})")
