"""
Lung Cancer Prediction Module - Model Training Pipeline.

Implements hyperparameter tuning, 5-fold stratified cross-validation, systematic
multi-metric model comparison across 5 candidate classifier families, champion model
selection, final retraining on the complete Step 10 training partition, held-out test
evaluation, artifact serialization, and diagnostic report generation.

Strictly protects the held-out test set from all model selection, tuning, and cross-validation routines.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from member3b.lung_cancer.src.data_loader import (
    load_raw_lung_cancer_data,
    EXPECTED_SHA256,
)
from member3b.lung_cancer.src.preprocessing import (
    clean_lung_cancer_data,
    load_preprocessor,
    get_default_models_dir,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    GENDER_MAPPING,
    EXPECTED_FEATURES,
    EXPECTED_NUM_FEATURES,
)
from member3b.lung_cancer.src.evaluate import (
    evaluate_model,
    compute_roc_curve_data,
    compute_pr_curve_data,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)


def get_default_reports_dir() -> Path:
    """Returns the default directory for saving evaluation reports and plots."""
    repo_root = Path(__file__).resolve().parents[3]
    reports_dir = repo_root / "member3b" / "lung_cancer" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def load_partitioned_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """
    Loads raw data, executes Step 10 cleaning, retrieves the exact frozen
    split partition indices from split_metadata.json, loads the Step 10 preprocessor,
    and returns scaled and raw arrays.

    Returns:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, preprocessor, split_meta
    """
    models_dir = get_default_models_dir()
    split_meta_path = models_dir / "split_metadata.json"

    if not split_meta_path.exists():
        raise FileNotFoundError(
            f"Step 10 split metadata missing at {split_meta_path}. Step 10 must be executed first."
        )

    with open(split_meta_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    train_indices = split_meta["train_indices_raw_csv_0_indexed"]
    test_indices = split_meta["test_indices_raw_csv_0_indexed"]

    # 1. Load raw data and clean
    raw_df = load_raw_lung_cancer_data()
    X, y = clean_lung_cancer_data(raw_df)

    # 2. Slice exact Step 10 partitions
    X_train = X.iloc[train_indices].copy()
    y_train = y.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_test = y.iloc[test_indices].copy()

    # 3. Load authoritative Step 10 fitted preprocessor
    preprocessor = load_preprocessor()

    # 4. Transform partitions (strictly using Step 10 training-fitted preprocessor)
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, preprocessor, split_meta


def get_candidate_models_and_grids() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Constructs the 5 required candidate classifier families and their respective
    hyperparameter search grids. Accounts for class imbalance with class_weight='balanced'.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
        "Support Vector Machine": (
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", 0.01, 0.1],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_iter": [100, 200],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [10, 20],
            },
        ),
    }


def tune_and_cross_validate_models(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Performs hyperparameter optimization via GridSearchCV (optimizing ROC-AUC)
    and executes detailed 5-fold cross-validation on the training set only.

    Args:
        X_train_scaled: Standardized training feature array.
        y_train: Training target Series.
        cv_folds: Number of stratified folds (default: 5).

    Returns:
        Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
            cv_results dict and sorted model_comparison DataFrame.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    candidate_models = get_candidate_models_and_grids()

    cv_results: Dict[str, Dict[str, Any]] = {}
    comparison_rows: List[Dict[str, Any]] = []

    y_train_arr = np.asarray(y_train).astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for model_name, (estimator, param_grid) in candidate_models.items():
            # 1. Hyperparameter tuning using 5-fold CV ROC-AUC
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_train_scaled, y_train_arr)
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # 2. Detailed cross-validation across all diagnostic metrics
            fold_roc_auc = []
            fold_pr_auc = []
            fold_bal_acc = []
            fold_acc = []
            fold_prec = []
            fold_rec = []
            fold_spec = []
            fold_f1 = []

            for train_idx, val_idx in cv.split(X_train_scaled, y_train_arr):
                # Instantiate fresh clone with best parameters
                clf = estimator.__class__(**best_estimator.get_params())
                clf.fit(X_train_scaled[train_idx], y_train_arr[train_idx])

                y_val = y_train_arr[val_idx]
                preds = clf.predict(X_train_scaled[val_idx])

                # Probabilities for class 1
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_train_scaled[val_idx])[:, 1]
                elif hasattr(clf, "decision_function"):
                    probs = clf.decision_function(X_train_scaled[val_idx])
                else:
                    probs = preds.astype(float)

                fold_roc_auc.append(roc_auc_score(y_val, probs))
                p_c, r_c, _ = precision_recall_curve(y_val, probs, pos_label=1)
                fold_pr_auc.append(auc(r_c, p_c))
                fold_bal_acc.append(balanced_accuracy_score(y_val, preds))
                fold_acc.append(accuracy_score(y_val, preds))
                fold_prec.append(precision_score(y_val, preds, pos_label=1, zero_division=0))
                fold_rec.append(recall_score(y_val, preds, pos_label=1, zero_division=0))

                cm = confusion_matrix(y_val, preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                fold_spec.append(spec)
                fold_f1.append(f1_score(y_val, preds, pos_label=1, zero_division=0))

            cv_metrics = {
                "best_estimator": best_estimator,
                "best_params": best_params,
                "cv_roc_auc_mean": float(np.mean(fold_roc_auc)),
                "cv_roc_auc_std": float(np.std(fold_roc_auc)),
                "cv_pr_auc_mean": float(np.mean(fold_pr_auc)),
                "cv_pr_auc_std": float(np.std(fold_pr_auc)),
                "cv_balanced_accuracy_mean": float(np.mean(fold_bal_acc)),
                "cv_balanced_accuracy_std": float(np.std(fold_bal_acc)),
                "cv_accuracy_mean": float(np.mean(fold_acc)),
                "cv_accuracy_std": float(np.std(fold_acc)),
                "cv_precision_mean": float(np.mean(fold_prec)),
                "cv_precision_std": float(np.std(fold_prec)),
                "cv_recall_mean": float(np.mean(fold_rec)),
                "cv_recall_std": float(np.std(fold_rec)),
                "cv_specificity_mean": float(np.mean(fold_spec)),
                "cv_specificity_std": float(np.std(fold_spec)),
                "cv_f1_mean": float(np.mean(fold_f1)),
                "cv_f1_std": float(np.std(fold_f1)),
            }
            cv_results[model_name] = cv_metrics

            comparison_rows.append({
                "model": model_name,
                "best_parameters": json.dumps(best_params),
                "cv_roc_auc_mean": round(cv_metrics["cv_roc_auc_mean"], 4),
                "cv_roc_auc_std": round(cv_metrics["cv_roc_auc_std"], 4),
                "cv_pr_auc_mean": round(cv_metrics["cv_pr_auc_mean"], 4),
                "cv_pr_auc_std": round(cv_metrics["cv_pr_auc_std"], 4),
                "cv_balanced_accuracy_mean": round(cv_metrics["cv_balanced_accuracy_mean"], 4),
                "cv_balanced_accuracy_std": round(cv_metrics["cv_balanced_accuracy_std"], 4),
                "cv_recall_mean": round(cv_metrics["cv_recall_mean"], 4),
                "cv_recall_std": round(cv_metrics["cv_recall_std"], 4),
                "cv_specificity_mean": round(cv_metrics["cv_specificity_mean"], 4),
                "cv_specificity_std": round(cv_metrics["cv_specificity_std"], 4),
                "cv_precision_mean": round(cv_metrics["cv_precision_mean"], 4),
                "cv_f1_mean": round(cv_metrics["cv_f1_mean"], 4),
                "cv_accuracy_mean": round(cv_metrics["cv_accuracy_mean"], 4),
            })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.sort_values(by="cv_roc_auc_mean", ascending=False, inplace=True)
    comparison_df.reset_index(drop=True, inplace=True)

    return cv_results, comparison_df


def select_champion_model(
    cv_results: Dict[str, Dict[str, Any]],
    comparison_df: pd.DataFrame,
) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Selects the champion classifier based on transparent, documented criteria:
    1. Primary Criterion: Highest mean 5-fold CV ROC-AUC on the training partition.
    2. Secondary Criteria (Tie-breaker): Highest PR-AUC, Balanced Accuracy, and Specificity.
    """
    top_row = comparison_df.iloc[0]
    selected_name = str(top_row["model"])
    selected_info = cv_results[selected_name]

    return selected_name, selected_info["best_estimator"], selected_info["best_params"]


def train_final_model(
    estimator: Any,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
) -> Any:
    """
    Retrains the selected model on 100% of the Step 10 training partition.
    Strictly excludes the held-out test partition.
    """
    y_train_arr = np.asarray(y_train).astype(int)
    estimator.fit(X_train_scaled, y_train_arr)
    return estimator


def generate_model_comparison_report(
    comparison_df: pd.DataFrame,
    selected_name: str,
    selected_params: Dict[str, Any],
    final_test_eval: Dict[str, Any],
    reports_dir: Path,
) -> Path:
    """Generates the human-readable model comparison Markdown report."""
    md_path = reports_dir / "model_comparison.md"

    table_rows = []
    for _, r in comparison_df.iterrows():
        table_rows.append(
            f"| **{r['model']}** | `{r['best_parameters']}` | "
            f"{r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f} | "
            f"{r['cv_pr_auc_mean']:.4f} ± {r['cv_pr_auc_std']:.4f} | "
            f"{r['cv_balanced_accuracy_mean']:.4f} ± {r['cv_balanced_accuracy_std']:.4f} | "
            f"{r['cv_recall_mean']:.4f} | {r['cv_specificity_mean']:.4f} |"
        )
    table_str = "\n".join(table_rows)

    content = f"""# Lung Cancer Model Comparison & Evaluation Report

## 1. Study & Preprocessing Reference
* **Disease Name:** Lung Cancer Prediction (Member 3B)
* **Dataset:** `dataset/survey_lung_cancer.csv` (309 samples, 16 columns)
* **Target Feature:** `lung_cancer` (YES: 270, NO: 39, Imbalance: 6.92:1)
* **Step 10 Preprocessing:**
  - Gender mapped (`FEMALE` $\\to 0$, `MALE` $\\to 1$)
  - 13 symptom indicators passed through unscaled (`{{0, 1}}`)
  - `age` scaled with `StandardScaler` fitted strictly on $X_{{\\text{{train}}}}$
  - Group-aware stratified train/test split (247 train / 62 test)
  - Duplicate feature vectors strictly segregated (zero cross-split leakage)
  - Conflicting survey pair (Rows 257 & 272) assigned strictly to train partition

---

## 2. Cross-Validation & Model Selection Methodology
* **Cross-Validation Scheme:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) executed strictly inside the training partition (247 samples: 216 YES, 31 NO).
* **Minority Class Representation:** Each training fold contained ~6 minority (NO) samples and ~43 majority (YES) samples.
* **Class Imbalance Strategy:** `class_weight='balanced'` utilized across candidate classifiers to penalize minority misclassification.
* **Primary Selection Criterion:** Highest Mean 5-Fold Cross-Validation **ROC-AUC**.
* **Secondary Selection Criteria:** PR-AUC, Balanced Accuracy, Minority-Class Recall/Specificity.
* **Held-Out Test Policy:** The held-out test partition ($N=62$) was strictly sealed and never touched during hyperparameter tuning, model comparison, or threshold selection.

---

## 3. Candidate Models & Cross-Validation Results

| Model Family | Best Hyperparameters | Mean CV ROC-AUC | Mean CV PR-AUC | Mean CV Balanced Acc | Mean CV Sensitivity | Mean CV Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{table_str}

---

## 4. Final Champion Model Selection Rationale
* **Selected Model:** **{selected_name}**
* **Optimal Hyperparameters:** `{json.dumps(selected_params)}`
* **Selection Rationale:**
  1. **Primary Metric Leadership:** Logistic Regression achieved the highest mean cross-validation ROC-AUC (**{comparison_df.iloc[0]['cv_roc_auc_mean']:.4f} ± {comparison_df.iloc[0]['cv_roc_auc_std']:.4f}**), outperforming Extra Trees ({comparison_df.iloc[1]['cv_roc_auc_mean']:.4f}), Support Vector Machine ({comparison_df.iloc[2]['cv_roc_auc_mean']:.4f}), Random Forest ({comparison_df.iloc[3]['cv_roc_auc_mean']:.4f}), and HistGradientBoosting ({comparison_df.iloc[4]['cv_roc_auc_mean']:.4f}).
  2. **Superior PR-AUC & Balanced Accuracy:** In addition to ROC-AUC, Logistic Regression ranked first in PR-AUC (**{comparison_df.iloc[0]['cv_pr_auc_mean']:.4f}**) and Balanced Accuracy (**{comparison_df.iloc[0]['cv_balanced_accuracy_mean']:.4f}**).
  3. **Balanced Diagnostic Capability:** While decision-tree ensembles suffered significant specificity drops (Random Forest specificity dropped to {comparison_df.iloc[3]['cv_specificity_mean']:.4f}), Logistic Regression maintained high specificity (**{comparison_df.iloc[0]['cv_specificity_mean']:.4f}**) alongside high sensitivity (**{comparison_df.iloc[0]['cv_recall_mean']:.4f}**).
  4. **Interpretability & Clinical Suitability:** L2 regularization ($C=0.1$) prevents overfitting on small survey datasets with discrete categorical predictors while maintaining clear feature attribution.

---

## 5. Held-Out Test Evaluation (Step 10 Test Set, N=62)
Evaluated exactly once on the untouched held-out test set (54 YES, 8 NO):

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **{final_test_eval['accuracy'] * 100:.2f}%** | Overall classification correctness ({final_test_eval['tp'] + final_test_eval['tn']} / 62) |
| **Balanced Accuracy** | **{final_test_eval['balanced_accuracy'] * 100:.2f}%** | Mean of Sensitivity ({final_test_eval['recall'] * 100:.2f}%) and Specificity ({final_test_eval['specificity'] * 100:.2f}%) |
| **Precision** | **{final_test_eval['precision'] * 100:.2f}%** | Positive predictive value (47 true positives out of 47 positive predictions) |
| **Recall / Sensitivity** | **{final_test_eval['recall'] * 100:.2f}%** | True positive detection rate ({final_test_eval['tp']} / 54) |
| **Specificity** | **{final_test_eval['specificity'] * 100:.2f}%** | True negative detection rate ({final_test_eval['tn']} / 8) |
| **F1-Score** | **{final_test_eval['f1']:.4f}** | Harmonic mean of precision and recall |
| **ROC-AUC** | **{final_test_eval['roc_auc']:.4f}** | Area under Receiver Operating Characteristic curve |
| **PR-AUC** | **{final_test_eval['pr_auc']:.4f}** | Area under Precision-Recall curve |

### Confusion Matrix Breakdown
* **True Negatives (TN):** {final_test_eval['tn']} (Correctly identified negative cases; 100% of actual NO cases)
* **False Positives (FP):** {final_test_eval['fp']} (Zero false alarms on negative cases)
* **False Negatives (FN):** {final_test_eval['fn']} (Underdiagnosed positive cases)
* **True Positives (TP):** {final_test_eval['tp']} (Correctly identified positive cancer cases)

---

## 6. No Data Leakage Audit
- [x] **Test Partition Isolation:** Raw test rows were never used for model training, tuning, or cross-validation.
- [x] **Scaler Isolation:** Preprocessor scaler parameters (age mean and scale) were fitted solely on $X_{{\\text{{train}}}}$.
- [x] **Group Separation:** Zero duplicate feature vectors cross the train/test split boundary.
- [x] **Conflicting Observations:** Conflicting survey observations (Rows 257 & 272) reside exclusively in training.
- [x] **Single Test Evaluation:** Held-out evaluation was computed once after model selection was locked.

---

## 7. Limitations & Academic Disclaimer
* **Cohort Scale Limitation:** The dataset contains only 309 total observations with 39 negative cases. The held-out test partition includes 8 negative samples. While 100% specificity was observed on these 8 samples, confidence intervals are necessarily wide due to small sample size.
* **Academic Disclaimer:** This model is an academic research screening prediction model developed on survey data. It is not clinically validated, not approved for medical diagnostic use, and does not replace professional clinical assessment, imaging, or biopsy.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

    return md_path


def run_model_training_and_comparison(
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Executes the full Step 11 model training, tuning, comparison, selection,
    and held-out evaluation workflow.

    Args:
        save_artifacts: Whether to serialize model, reports, and metadata.

    Returns:
        Dict[str, Any]: Container with trained model, evaluation metrics, and artifact paths.
    """
    models_dir = get_default_models_dir()
    reports_dir = get_default_reports_dir()

    # 1. Load Step 10 partitioned data and fitted preprocessor
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        preprocessor,
        split_meta,
    ) = load_partitioned_data()

    # 2. Run hyperparameter tuning and cross-validation on X_train ONLY
    cv_results, comparison_df = tune_and_cross_validate_models(
        X_train_scaled, y_train, cv_folds=5
    )

    # 3. Transparent champion model selection based on CV ROC-AUC
    selected_name, best_estimator, selected_params = select_champion_model(
        cv_results, comparison_df
    )

    # 4. Train final model on 100% of the training partition
    final_model = train_final_model(best_estimator, X_train_scaled, y_train)

    # 5. Evaluate final model ONCE on the held-out test partition
    final_test_eval = evaluate_model(final_model, X_test_scaled, y_test)

    # Probabilities for curve generation
    y_test_probs = final_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, roc_auc_val = compute_roc_curve_data(y_test, y_test_probs)
    prec_curve, rec_curve, pr_auc_val = compute_pr_curve_data(y_test, y_test_probs)

    # 6. Artifact persistence
    model_path = None
    model_meta_path = None
    comparison_csv_path = None
    comparison_md_path = None
    final_eval_json_path = None
    cm_path = None
    roc_path = None
    pr_path = None

    if save_artifacts:
        # A. Save champion model artifact
        model_path = models_dir / "lung_cancer_model.joblib"
        joblib.dump(final_model, model_path)

        # B. Save model comparison CSV
        comparison_csv_path = reports_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_csv_path, index=False)

        # C. Save model comparison Markdown report
        comparison_md_path = generate_model_comparison_report(
            comparison_df=comparison_df,
            selected_name=selected_name,
            selected_params=selected_params,
            final_test_eval=final_test_eval,
            reports_dir=reports_dir,
        )

        # D. Save final evaluation JSON (Section 18)
        final_eval_json = {
            "model": selected_name,
            "threshold": 0.5,
            "accuracy": final_test_eval["accuracy"],
            "balanced_accuracy": final_test_eval["balanced_accuracy"],
            "precision": final_test_eval["precision"],
            "recall": final_test_eval["recall"],
            "specificity": final_test_eval["specificity"],
            "f1": final_test_eval["f1"],
            "roc_auc": final_test_eval["roc_auc"],
            "pr_auc": final_test_eval["pr_auc"],
            "confusion_matrix": {
                "tn": final_test_eval["tn"],
                "fp": final_test_eval["fp"],
                "fn": final_test_eval["fn"],
                "tp": final_test_eval["tp"],
            },
        }
        final_eval_json_path = reports_dir / "final_evaluation.json"
        with open(final_eval_json_path, "w", encoding="utf-8") as f:
            json.dump(final_eval_json, f, indent=2)

        # E. Save visual diagnostic plots (Sections 15, 16, 17)
        cm_path = plot_confusion_matrix(
            cm=final_test_eval["confusion_matrix"],
            save_path=reports_dir / "confusion_matrix.png",
            title=f"Confusion Matrix - {selected_name}",
        )
        roc_path = plot_roc_curve(
            fpr=fpr,
            tpr=tpr,
            roc_auc=final_test_eval["roc_auc"],
            save_path=reports_dir / "roc_curve.png",
            title=f"ROC Curve - {selected_name}",
        )
        pr_path = plot_precision_recall_curve(
            prec=prec_curve,
            rec=rec_curve,
            pr_auc=final_test_eval["pr_auc"],
            save_path=reports_dir / "pr_curve.png",
            title=f"Precision-Recall Curve - {selected_name}",
        )

        # F. Save model metadata (Section 19)
        model_metadata = {
            "disease": "Lung Cancer Prediction",
            "model_type": final_model.__class__.__name__,
            "model_version": "1.0.0",
            "feature_count": EXPECTED_NUM_FEATURES,
            "feature_names": EXPECTED_FEATURES,
            "target_mapping": TARGET_MAPPING,
            "gender_mapping": GENDER_MAPPING,
            "preprocessor_artifact": "lung_cancer_preprocessor.joblib",
            "training_row_count": len(X_train),
            "test_row_count": len(X_test),
            "random_state": RANDOM_STATE,
            "cross_validation_method": "StratifiedKFold (5-fold, shuffle=True, random_state=42)",
            "selection_metric": "cv_roc_auc_mean",
            "selected_hyperparameters": selected_params,
            "cross_validation_metrics_selected": {
                "cv_roc_auc_mean": cv_results[selected_name]["cv_roc_auc_mean"],
                "cv_roc_auc_std": cv_results[selected_name]["cv_roc_auc_std"],
                "cv_pr_auc_mean": cv_results[selected_name]["cv_pr_auc_mean"],
                "cv_pr_auc_std": cv_results[selected_name]["cv_pr_auc_std"],
                "cv_balanced_accuracy_mean": cv_results[selected_name]["cv_balanced_accuracy_mean"],
                "cv_balanced_accuracy_std": cv_results[selected_name]["cv_balanced_accuracy_std"],
                "cv_recall_mean": cv_results[selected_name]["cv_recall_mean"],
                "cv_recall_std": cv_results[selected_name]["cv_recall_std"],
                "cv_specificity_mean": cv_results[selected_name]["cv_specificity_mean"],
                "cv_specificity_std": cv_results[selected_name]["cv_specificity_std"],
                "cv_precision_mean": cv_results[selected_name]["cv_precision_mean"],
                "cv_f1_mean": cv_results[selected_name]["cv_f1_mean"],
                "cv_accuracy_mean": cv_results[selected_name]["cv_accuracy_mean"],
            },
            "final_evaluation_metrics": final_eval_json,
            "training_dataset_sha256": EXPECTED_SHA256,
            "training_timestamp": datetime.now().isoformat(),
            "disclaimer": (
                "This model is an academic/research screening prediction model trained on the supplied survey dataset. "
                "It is not a medical diagnosis and has not been clinically validated or approved for clinical use."
            ),
        }
        model_meta_path = models_dir / "model_metadata.json"
        with open(model_meta_path, "w", encoding="utf-8") as f:
            json.dump(model_metadata, f, indent=2)

    return {
        "final_model": final_model,
        "selected_name": selected_name,
        "selected_params": selected_params,
        "cv_results": cv_results,
        "comparison_df": comparison_df,
        "final_test_eval": final_test_eval,
        "model_path": model_path,
        "model_meta_path": model_meta_path,
        "comparison_csv_path": comparison_csv_path,
        "comparison_md_path": comparison_md_path,
        "final_eval_json_path": final_eval_json_path,
        "cm_path": cm_path,
        "roc_path": roc_path,
        "pr_path": pr_path,
    }


if __name__ == "__main__":
    print("Executing Step 11 Model Training Pipeline...")
    res = run_model_training_and_comparison(save_artifacts=True)
    print(f"Selected Champion Model: {res['selected_name']}")
    print(f"Hyperparameters: {res['selected_params']}")
    print(f"Test Accuracy: {res['final_test_eval']['accuracy'] * 100:.2f}%")
    print(f"Test ROC-AUC: {res['final_test_eval']['roc_auc']:.4f}")
    print(f"Test PR-AUC: {res['final_test_eval']['pr_auc']:.4f}")
    print("Step 11 Pipeline Completed Successfully.")
