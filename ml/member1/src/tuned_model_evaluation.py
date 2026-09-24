"""Evaluate tuned disease models on the untouched Phase 7 test split.

Model selection/tuning is assumed to have been performed only on the training
split. This module fits the selected tuned pipelines on X_train and evaluates
them once on X_test. Preprocessing stays inside each Pipeline.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def evaluate_models(
    X_train, X_test, y_train, y_test, preprocessor,
    model_specs, positive_class=None
):
    rows=[]
    details={}
    for name, (model, params) in model_specs.items():
        model.set_params(**params)
        pipe=Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred=pipe.predict(X_test)

        classes=getattr(pipe.named_steps["model"], "classes_", None)
        if hasattr(pipe, "predict_proba"):
            scores=pipe.predict_proba(X_test)
            if positive_class is None:
                positive_index=1
            else:
                positive_index=list(classes).index(positive_class)
            y_score=scores[:, positive_index]
        elif hasattr(pipe, "decision_function"):
            scores=pipe.decision_function(X_test)
            y_score=scores
        else:
            y_score=None

        if positive_class is not None:
            y_true_bin=(pd.Series(y_test).to_numpy()==positive_class).astype(int)
            y_pred_bin=(pd.Series(y_pred).to_numpy()==positive_class).astype(int)
        else:
            y_true_bin=pd.Series(y_test).to_numpy()
            y_pred_bin=pd.Series(y_pred).to_numpy()

        row={
            "model":name,
            "accuracy":accuracy_score(y_test,y_pred),
            "precision":precision_score(y_true_bin,y_pred_bin,zero_division=0),
            "recall":recall_score(y_true_bin,y_pred_bin,zero_division=0),
            "f1":f1_score(y_true_bin,y_pred_bin,zero_division=0),
            "roc_auc":roc_auc_score(y_true_bin,y_score) if y_score is not None else None,
            "pr_auc":average_precision_score(y_true_bin,y_score) if y_score is not None else None,
        }
        cm=confusion_matrix(y_true_bin,y_pred_bin).tolist()
        rows.append(row)
        details[name]={
            "params":params,
            "confusion_matrix":[[int(v) for v in row_] for row_ in cm],
        }
    return pd.DataFrame(rows), details

def save_results(results, details, csv_path, json_path):
    Path(csv_path).parent.mkdir(parents=True,exist_ok=True)
    results.to_csv(csv_path,index=False)
    Path(json_path).write_text(json.dumps(details,indent=2),encoding="utf-8")
