"""Step 1 dataset verification for Member 3A.

Run from the repository root:
    python scripts/verify_member3_datasets.py
"""

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASTHMA = ROOT / "dataset" / "synthetic_asthma_dataset.csv"
PARKINSONS = ROOT / "dataset" / "Parkinsons_Disease_Dataset.xlsx"


def summarize(df, target=None):
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for c in df.columns:
        print(f"  - {c}: {df[c].dtype}")

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    if target and target in df.columns:
        print(f"\nTarget: {target}")
        print(df[target].value_counts(dropna=False).to_string())
        print("\nTarget percentages:")
        print((df[target].value_counts(normalize=True, dropna=False) * 100).round(2).to_string())

    print("\nUnique values:")
    print(df.nunique(dropna=False).sort_values().to_string())

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print(f"\nConstant columns: {constant or 'None'}")


def verify_asthma():
    print("=" * 80)
    print("ASTHMA DATASET")
    print("=" * 80)
    if not ASTHMA.exists():
        raise FileNotFoundError(ASTHMA)
    df = pd.read_csv(ASTHMA)
    summarize(df, "Has_Asthma")

    if "Patient_ID" in df:
        print(f"\nPatient_ID unique: {df['Patient_ID'].nunique()} / {len(df)}")

    if "Asthma_Control_Level" in df:
        print("\nAsthma_Control_Level values:")
        print(df["Asthma_Control_Level"].value_counts(dropna=False).to_string())


def verify_parkinsons():
    print("=" * 80)
    print("PARKINSON'S DATASET")
    print("=" * 80)
    if not PARKINSONS.exists():
        raise FileNotFoundError(PARKINSONS)

    df = pd.read_excel(PARKINSONS)
    summarize(df, "status")

    if "name" in df.columns:
        names = df["name"].astype(str)
        subject_matches = names.str.extract(r"(S\d+)", expand=False)
        print(f"\nUnique name values: {df['name'].nunique()}")
        print(f"Rows with extracted S-number subject ID: {subject_matches.notna().sum()}")
        print(f"Unique extracted subject IDs: {subject_matches.nunique(dropna=True)}")
        print("\nRecordings per extracted subject (top 20):")
        print(subject_matches.value_counts().head(20).to_string())

        if "status" in df.columns:
            print("\nSubject/name vs target crosstab:")
            print(pd.crosstab(names, df["status"]).head(20).to_string())


if __name__ == "__main__":
    verify_asthma()
    verify_parkinsons()
    print("\nStep 1 verification finished.")
