"""Shared candidate-model factory for Member 1.

Phase 10: defines comparable classification candidates.
No final model selection is performed here.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_models(random_state: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def add_xgboost(models: dict, random_state: int = 42) -> dict:
    """Add XGBoost only when the optional dependency is available."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return models

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    return models
