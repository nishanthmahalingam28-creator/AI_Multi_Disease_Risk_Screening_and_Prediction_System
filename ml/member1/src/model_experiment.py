"""Reusable Phase 10/11 model experiment runner.

The estimator is always wrapped with the disease-specific preprocessor so that
preprocessing is fitted independently inside each cross-validation fold.
The untouched test set is not used by this module for model selection.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

MEMBER1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBER1_DIR))

from src.model_candidates import build_models, add_xgboost

RANDOM_STATE = 42
N_SPLITS = 5


def run_model_experiments(X_train, y_train, preprocessor, include_xgboost=True):
    models = build_models(RANDOM_STATE)
    if include_xgboost:
        models = add_xgboost(models, RANDOM_STATE)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    rows = []
    fitted_pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        rows.append({
            "model": name,
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_precision_mean": scores["test_precision"].mean(),
            "cv_precision_std": scores["test_precision"].std(),
            "cv_recall_mean": scores["test_recall"].mean(),
            "cv_recall_std": scores["test_recall"].std(),
            "cv_f1_mean": scores["test_f1"].mean(),
            "cv_f1_std": scores["test_f1"].std(),
            "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
            "cv_roc_auc_std": scores["test_roc_auc"].std(),
        })
        fitted_pipelines[name] = pipeline

    return pd.DataFrame(rows), fitted_pipelines
