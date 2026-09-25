"""Leakage-safe preprocessing pipeline for Heart Disease.

Phase 6 only: this module constructs an unfitted scikit-learn preprocessor.
The caller is responsible for fitting it only on training data/folds.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
CATEGORICAL_FEATURES = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
]
TARGET_COLUMN = "Heart Disease"


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted Heart Disease preprocessing transformer."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")
