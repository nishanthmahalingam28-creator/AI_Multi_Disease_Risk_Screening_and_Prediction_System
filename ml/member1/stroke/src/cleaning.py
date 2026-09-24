"""
Stroke Data Cleaning & Leakage Prevention Module.

Responsibilities:
- Ingest raw DataFrame and perform deterministic, reproducible data cleaning.
- Standardize column names (strip leading/trailing whitespace).
- Exclude patient primary key ('id') from the ML dataset to prevent acute target leakage.
- Strictly preserve missing BMI entries (201 records) as NaN (DO NOT delete rows, as 16% of stroke cases lack BMI).
- Preserve informative categorical values:
    - 'smoking_status': 'Unknown' represents uncollected history; retained as valid categorical level.
    - 'gender': Rare category 'Other' (1 record) is retained.
- Target column 'stroke' validated as binary integer (0, 1).
- Return a fresh cleaned DataFrame copy without mutating the input.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Deterministic Target Class Mapping
STROKE_TARGET_MAPPING = {
    0: 0,
    1: 1
}


def clean_stroke_data(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """
    Cleans the raw Stroke dataset deterministically.
    Removes patient identifier 'id' to prevent data leakage and preserves missing BMI entries.

    Args:
        df (pd.DataFrame): Raw DataFrame loaded from dataset/stroke.csv.
        drop_id (bool): Whether to drop the administrative 'id' column. Defaults to True.

    Returns:
        pd.DataFrame: Cleaned working DataFrame ready for leak-free pipeline preprocessing.
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
        print(f"[Stroke Cleaning] Removed {duplicates_removed} duplicate rows.")

    # 4. Remove administrative identifier 'id' to prevent data leakage
    if drop_id and "id" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["id"])

    # 5. Clean string/categorical entries (strip surrounding whitespace)
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for col in cat_cols:
        if col in cleaned_df.columns and cleaned_df[col].dtype == object:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # 6. Target validation
    target_col = "stroke"
    if target_col in cleaned_df.columns:
        cleaned_df[target_col] = cleaned_df[target_col].astype(int)
        if not set(cleaned_df[target_col].unique()).issubset({0, 1}):
            raise ValueError("[Stroke Cleaning] Unexpected target classes encountered in 'stroke'!")

    return cleaned_df


if __name__ == "__main__":
    from ml.member1.config import STROKE_RAW_DATA_PATH, get_disease_paths

    paths = get_disease_paths("stroke")
    raw_df = pd.read_csv(STROKE_RAW_DATA_PATH)
    cleaned = clean_stroke_data(raw_df)

    output_path = paths["data"] / "cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"[Stroke Cleaning] Cleaned dataset saved to {output_path} (Shape: {cleaned.shape})")
    print(f"[Stroke Cleaning] Columns present: {list(cleaned.columns)}")
    print(f"[Stroke Cleaning] Missing values preserved in bmi: {cleaned['bmi'].isnull().sum()}")
