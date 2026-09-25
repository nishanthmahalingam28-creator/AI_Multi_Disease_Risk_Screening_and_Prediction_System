"""
Breast Cancer Prediction Module - Training Pipeline.

Implements model training, stratified cross-validation, hyperparameter tuning,
systematic model comparison, and serialization of the selected champion classifier.
Strictly isolates the test partition from all hyperparameter selection and CV routines.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)

from member3b.breast_cancer.src.preprocessing import (
    prepare_breast_cancer_data,
    RANDOM_STATE,
)
from member3b.breast_cancer.src.evaluate import (
    evaluate_model,
    compute_roc_curve_data,
    compute_pr_curve_data,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_model_comparison,
)


def get_default_models_dir() -> Path:
    """Returns the default directory for saving model artifacts."""
    repo_root = Path(__file__).resolve().parents[3]
    models_dir = repo_root / "member3b" / "breast_cancer" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_default_reports_dir() -> Path:
    """Returns the default directory for saving model evaluation reports."""
    repo_root = Path(__file__).resolve().parents[3]
    reports_dir = repo_root / "member3b" / "breast_cancer" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def get_candidate_models_and_grids() -> Dict[str, Tuple[Any, List[Dict[str, Any]]]]:
    """
    Defines the 5 required candidate classifier families and their respective
    hyperparameter search grids. Compact and medically appropriate.
    """
    models_and_grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            [
                {"penalty": ["l2"], "C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs"]},
                {"penalty": ["l1"], "C": [0.01, 0.1, 1.0, 10.0], "solver": ["saga"]},
            ],
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=RANDOM_STATE),
            [
                {"kernel": ["rbf"], "C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto", 0.01]},
                {"kernel": ["linear"], "C": [0.01, 0.1, 1.0, 10.0]},
            ],
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            [
                {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2"],
                }
            ],
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            [
                {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 4],
                    "subsample": [0.8, 1.0],
                }
            ],
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE),
            [
                {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2"],
                }
            ],
        ),
    }
    return models_and_grids


def tune_and_cross_validate_models(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Performs hyperparameter tuning and cross-validation on the training set only.

    Args:
        X_train_scaled: Standardized training feature array.
        y_train: Training target labels.
        cv_folds: Number of stratified folds (default: 5).

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of tuned candidate models and their CV performance.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    models_and_grids = get_candidate_models_and_grids()
    results: Dict[str, Dict[str, Any]] = {}

    scoring_metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    for model_name, (estimator, param_grid) in models_and_grids.items():
        # 1. Hyperparameter tuning using ROC-AUC
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train_scaled, y_train)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # 2. Detailed cross-validation of the best estimator across all metrics
        cv_scores = cross_validate(
            estimator=best_estimator,
            X=X_train_scaled,
            y=y_train,
            cv=cv,
            scoring=scoring_metrics,
            return_train_score=False,
            n_jobs=-1,
        )

        results[model_name] = {
            "best_estimator": best_estimator,
            "best_params": best_params,
            "best_cv_roc_auc": float(grid_search.best_score_),
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
            "cv_precision_std": float(np.std(cv_scores["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
            "cv_recall_std": float(np.std(cv_scores["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
            "cv_f1_std": float(np.std(cv_scores["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        }

    return results


def run_model_training_and_comparison(
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Executes the complete Step 4 training, tuning, comparison, and evaluation pipeline.

    Args:
        save_artifacts: Whether to save model, reports, and metadata.

    Returns:
        Dict[str, Any]: Summary dictionary containing all experiment results and artifact paths.
    """
    # 1. Prepare data using Step 3 pipeline
    prep_data = prepare_breast_cancer_data(save_artifacts=save_artifacts)
    X_train_scaled = prep_data["X_train_scaled"]
    X_test_scaled = prep_data["X_test_scaled"]
    y_train = np.asarray(prep_data["y_train"]).astype(int)
    y_test = np.asarray(prep_data["y_test"]).astype(int)
    feature_names = prep_data["metadata"]["feature_names"]

    # 2. Run hyperparameter tuning & cross-validation on training data ONLY
    cv_results = tune_and_cross_validate_models(X_train_scaled, y_train, cv_folds=5)

    # 3. Model comparison table construction
    comparison_rows = []
    trained_models = {}

    for model_name, res in cv_results.items():
        # Fit tuned model on full training set
        model = res["best_estimator"]
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model

        # Evaluate on the held-out test set
        test_eval = evaluate_model(model, X_test_scaled, y_test)

        comparison_rows.append({
            "model": model_name,
            "cv_accuracy_mean": round(res["cv_accuracy_mean"], 4),
            "cv_accuracy_std": round(res["cv_accuracy_std"], 4),
            "cv_precision_mean": round(res["cv_precision_mean"], 4),
            "cv_precision_std": round(res["cv_precision_std"], 4),
            "cv_recall_mean": round(res["cv_recall_mean"], 4),
            "cv_recall_std": round(res["cv_recall_std"], 4),
            "cv_f1_mean": round(res["cv_f1_mean"], 4),
            "cv_f1_std": round(res["cv_f1_std"], 4),
            "cv_roc_auc_mean": round(res["cv_roc_auc_mean"], 4),
            "cv_roc_auc_std": round(res["cv_roc_auc_std"], 4),
            "test_accuracy": round(test_eval["accuracy"], 4),
            "test_precision": round(test_eval["precision"], 4),
            "test_recall": round(test_eval["recall"], 4),
            "test_specificity": round(test_eval["specificity"], 4),
            "test_f1": round(test_eval["f1"], 4),
            "test_roc_auc": round(test_eval["roc_auc"], 4) if test_eval["roc_auc"] else None,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    # Sort primarily by cv_roc_auc_mean descending
    comparison_df.sort_values(by="cv_roc_auc_mean", ascending=False, inplace=True)

    # 4. Primary Model Selection based on training CV ROC-AUC
    selected_model_name = str(comparison_df.iloc[0]["model"])
    selected_model = trained_models[selected_model_name]
    selected_params = cv_results[selected_model_name]["best_params"]

    # 5. Final Evaluation for the selected model
    final_test_eval = evaluate_model(selected_model, X_test_scaled, y_test)

    # Compute ROC and PR curve data for selected model
    y_test_probs = selected_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, roc_auc_val = compute_roc_curve_data(y_test, y_test_probs)
    pr_prec, pr_rec, pr_auc_val = compute_pr_curve_data(y_test, y_test_probs)

    # 6. Save Artifacts & Reports
    models_dir = get_default_models_dir()
    reports_dir = get_default_reports_dir()

    model_path = None
    metadata_path = None
    comparison_csv_path = None

    if save_artifacts:
        # Save champion model
        model_path = models_dir / "breast_cancer_model.joblib"
        joblib.dump(selected_model, model_path)

        # Save comparison CSV
        comparison_csv_path = reports_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_csv_path, index=False)

        # Generate Visualizations
        plot_model_comparison(
            comparison_df=comparison_df,
            save_path=reports_dir / "model_comparison_chart.png",
        )
        plot_confusion_matrix(
            cm=final_test_eval["confusion_matrix"],
            save_path=reports_dir / "confusion_matrix.png",
            title=f"Confusion Matrix - {selected_model_name}",
        )
        plot_roc_curve(
            fpr=fpr,
            tpr=tpr,
            roc_auc=final_test_eval["roc_auc"],
            save_path=reports_dir / "roc_curve.png",
            title=f"ROC Curve - {selected_model_name}",
        )
        plot_precision_recall_curve(
            prec=pr_prec,
            rec=pr_rec,
            pr_auc=pr_auc_val,
            save_path=reports_dir / "precision_recall_curve.png",
            title=f"Precision-Recall Curve - {selected_model_name}",
        )

        # Compile comprehensive real metadata
        metadata = {
            "disease_name": "Breast Cancer Prediction",
            "module_owner": "Member 3B",
            "target_column": "diagnosis",
            "target_mapping": {"B": 0, "M": 1},
            "positive_class": "1 (Malignant)",
            "negative_class": "0 (Benign)",
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "preprocessing_method": "StandardScaler (fitted strictly on X_train)",
            "random_state": RANDOM_STATE,
            "train_test_split": {"train_ratio": 0.80, "test_ratio": 0.20},
            "cross_validation": {
                "method": "StratifiedKFold",
                "folds": 5,
                "shuffle": True,
                "random_state": RANDOM_STATE,
            },
            "primary_selection_metric": "cv_roc_auc_mean",
            "models_evaluated": list(cv_results.keys()),
            "xgboost_status": "Not included; no verified compatible wheel available for Python 3.14 in this environment.",
            "hyperparameter_search_method": "GridSearchCV(scoring='roc_auc', cv=5)",
            "selected_model_name": selected_model_name,
            "selected_hyperparameters": selected_params,
            "cross_validation_metrics_selected": {
                "cv_accuracy_mean": cv_results[selected_model_name]["cv_accuracy_mean"],
                "cv_accuracy_std": cv_results[selected_model_name]["cv_accuracy_std"],
                "cv_precision_mean": cv_results[selected_model_name]["cv_precision_mean"],
                "cv_precision_std": cv_results[selected_model_name]["cv_precision_std"],
                "cv_recall_mean": cv_results[selected_model_name]["cv_recall_mean"],
                "cv_recall_std": cv_results[selected_model_name]["cv_recall_std"],
                "cv_f1_mean": cv_results[selected_model_name]["cv_f1_mean"],
                "cv_f1_std": cv_results[selected_model_name]["cv_f1_std"],
                "cv_roc_auc_mean": cv_results[selected_model_name]["cv_roc_auc_mean"],
                "cv_roc_auc_std": cv_results[selected_model_name]["cv_roc_auc_std"],
            },
            "final_test_metrics_selected": {
                "accuracy": final_test_eval["accuracy"],
                "precision": final_test_eval["precision"],
                "recall_sensitivity": final_test_eval["recall"],
                "specificity": final_test_eval["specificity"],
                "f1": final_test_eval["f1"],
                "roc_auc": final_test_eval["roc_auc"],
                "pr_auc": round(pr_auc_val, 4),
            },
            "confusion_matrix_selected": {
                "matrix": final_test_eval["confusion_matrix"],
                "tn": final_test_eval["tn"],
                "fp": final_test_eval["fp"],
                "fn": final_test_eval["fn"],
                "tp": final_test_eval["tp"],
            },
            "decision_threshold": 0.50,
            "threshold_policy": "Default classifier threshold (0.5); no clinical calibration applied.",
            "model_artifact_path": str(model_path),
            "preprocessor_artifact_path": str(prep_data["preprocessor_path"]),
            "training_timestamp": datetime.now().isoformat(),
            "disclaimer": (
                "This model is developed for an academic machine learning research project. "
                "It is NOT a clinically validated diagnostic tool and must not be used as a substitute "
                "for professional medical diagnosis or clinical decision making."
            ),
        }

        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {
        "comparison_df": comparison_df,
        "selected_model_name": selected_model_name,
        "selected_params": selected_params,
        "final_test_eval": final_test_eval,
        "cv_results": cv_results,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "comparison_csv_path": comparison_csv_path,
    }
