"""Leakage-safe preprocessing pipeline for Stroke.

Phase 6 only: this module constructs an unfitted scikit-learn preprocessor.
The caller is responsible for fitting it only on training data/folds.
The source identifier id is intentionally excluded.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = [
    "gender", "ever_married", "work_type", "Residence_type", "smoking_status",
]
EXCLUDED_FEATURES = ["id"]
TARGET_COLUMN = "stroke"


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted Stroke preprocessing transformer."""
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
