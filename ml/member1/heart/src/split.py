"""Heart Disease train/test split for Phase 7."""

from pathlib import Path
import sys
import pandas as pd

MEMBER1_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MEMBER1_DIR.parent.parent.parent
sys.path.insert(0, str(MEMBER1_DIR))

from src.split_data import create_train_test_split

DATA_PATH = PROJECT_ROOT / "ml/member1/heart/data/cleaned.csv"
TARGET_COLUMN = "Heart Disease"


def load_and_split(test_size: float = 0.20, random_state: int = 42):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return create_train_test_split(X, y, test_size, random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Heart Disease: train={len(X_train)}, test={len(X_test)}")
    print("Train distribution:")
    print(y_train.value_counts().sort_index())
    print("Test distribution:")
    print(y_test.value_counts().sort_index())
