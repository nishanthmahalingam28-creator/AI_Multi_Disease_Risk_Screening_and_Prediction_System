"""Phase 7 train/test split validation."""

from pathlib import Path
import sys
import pandas as pd

MEMBER1_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MEMBER1_DIR.parent.parent
sys.path.insert(0, str(MEMBER1_DIR))

from src.split_data import create_train_test_split
from heart.src.split import load_and_split as split_heart
from diabetes.src.split import load_and_split as split_diabetes
from stroke.src.split import load_and_split as split_stroke


def _check(loader, total, disease):
    X_train, X_test, y_train, y_test = loader()
    assert len(X_train) + len(X_test) == total
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) == round(total * 0.8)
    assert len(X_test) == total - round(total * 0.8)
    assert y_train.value_counts(normalize=True).to_dict()
    assert y_test.value_counts(normalize=True).to_dict()
    if disease == "stroke":
        assert "id" not in X_train.columns
        assert "id" not in X_test.columns


def test_heart_split():
    _check(split_heart, 270, "heart")


def test_diabetes_split():
    _check(split_diabetes, 768, "diabetes")


def test_stroke_split():
    _check(split_stroke, 5110, "stroke")


def test_reproducibility():
    df = pd.read_csv(PROJECT_ROOT / "ml/member1/heart/data/cleaned.csv")
    X = df.drop(columns=["Heart Disease"])
    y = df["Heart Disease"]
    a = create_train_test_split(X, y)
    b = create_train_test_split(X, y)
    assert a[0].index.equals(b[0].index)
    assert a[1].index.equals(b[1].index)


def test_different_seed_changes_split():
    df = pd.read_csv(PROJECT_ROOT / "ml/member1/heart/data/cleaned.csv")
    X = df.drop(columns=["Heart Disease"])
    y = df["Heart Disease"]
    a = create_train_test_split(X, y, random_state=42)
    b = create_train_test_split(X, y, random_state=7)
    assert not a[0].index.equals(b[0].index)
