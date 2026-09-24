"""Leakage-safe preprocessing pipeline for Diabetes.

Phase 6 only: this module constructs an unfitted scikit-learn preprocessor.
The caller is responsible for fitting it only on training data/folds.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERICAL_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
CATEGORICAL_FEATURES: list[str] = []
TARGET_COLUMN = "Outcome"


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted Diabetes preprocessing transformer."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERICAL_FEATURES),
    ], remainder="drop")
