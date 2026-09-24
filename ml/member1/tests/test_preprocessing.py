"""Phase 6 preprocessing smoke tests; no model training is performed."""

from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

MEMBER1_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MEMBER1_DIR.parent.parent
sys.path.insert(0, str(MEMBER1_DIR))

from heart.src.preprocessing import CATEGORICAL_FEATURES as HEART_CAT, NUMERICAL_FEATURES as HEART_NUM, TARGET_COLUMN as HEART_TARGET, build_preprocessor as build_heart_preprocessor
from diabetes.src.preprocessing import NUMERICAL_FEATURES as DIABETES_NUM, TARGET_COLUMN as DIABETES_TARGET, build_preprocessor as build_diabetes_preprocessor
from stroke.src.preprocessing import CATEGORICAL_FEATURES as STROKE_CAT, NUMERICAL_FEATURES as STROKE_NUM, TARGET_COLUMN as STROKE_TARGET, build_preprocessor as build_stroke_preprocessor


def _check(df, target, numeric, categorical, build_preprocessor, disease):
    X = df.drop(columns=[target])
    y = df[target]
    assert target not in numeric + categorical
    assert set(numeric + categorical) == set(X.columns)
    if disease == "stroke":
        assert "id" not in X.columns
        assert "id" not in numeric + categorical

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    assert X_train_t.shape[1] == X_test_t.shape[1]
    assert X_train_t.shape[0] == len(X_train)
    assert X_test_t.shape[0] == len(X_test)
    assert len(preprocessor.get_feature_names_out()) == X_train_t.shape[1]


def test_heart_preprocessing():
    df = pd.read_csv(PROJECT_ROOT / "ml/member1/heart/data/cleaned.csv")
    _check(df, HEART_TARGET, HEART_NUM, HEART_CAT, build_heart_preprocessor, "heart")


def test_diabetes_preprocessing():
    df = pd.read_csv(PROJECT_ROOT / "ml/member1/diabetes/data/cleaned.csv")
    _check(df, DIABETES_TARGET, DIABETES_NUM, [], build_diabetes_preprocessor, "diabetes")


def test_stroke_preprocessing():
    df = pd.read_csv(PROJECT_ROOT / "ml/member1/stroke/data/cleaned.csv")
    assert "id" not in df.columns
    _check(df, STROKE_TARGET, STROKE_NUM, STROKE_CAT, build_stroke_preprocessor, "stroke")
