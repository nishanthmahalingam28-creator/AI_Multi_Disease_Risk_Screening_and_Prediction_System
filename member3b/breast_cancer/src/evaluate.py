"""
Breast Cancer Prediction Module - Evaluation and Validation.

Implements reusable metric calculations and diagnostic visualization routines.
Specifically computes medical-grade diagnostic indicators including Sensitivity,
Specificity, Precision, F1-Score, ROC-AUC, and Confusion Matrix.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)


def calculate_classification_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_prob: Optional[Union[np.ndarray, pd.Series, list]] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics for binary diagnosis.

    Target definitions:
        Positive class (1): Malignant
        Negative class (0): Benign

    Args:
        y_true: Ground truth binary target vector.
        y_pred: Predicted discrete labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (1).

    Returns:
        Dict[str, Any]: Dictionary containing all computed metric values.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    # Core scores
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    # Confusion matrix extraction: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Medical sensitivity (Recall) and Specificity
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC if probabilities provided
    roc_auc_val: Optional[float] = None
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        roc_auc_val = float(roc_auc_score(y_true_arr, y_prob_arr))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc_val,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on a given feature matrix and target.

    Args:
        model: Trained estimator with predict() and optionally predict_proba().
        X: Feature matrix.
        y: True binary labels.

    Returns:
        Dict[str, Any]: Evaluation metrics dictionary.
    """
    y_pred = model.predict(X)

    y_prob: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] > 1:
            y_prob = probs[:, 1]
    elif hasattr(model, "decision_function"):
        # For classifiers like LinearSVC without predict_proba
        y_prob = model.decision_function(X)

    return calculate_classification_metrics(y_true=y, y_pred=y_pred, y_prob=y_prob)


def compute_roc_curve_data(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes ROC curve FPR, TPR, and area under the curve.

    Args:
        y_true: True binary labels.
        y_prob: Predicted positive class probabilities.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: fpr, tpr, roc_auc
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, roc_auc


def compute_pr_curve_data(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes Precision-Recall curve data and PR-AUC.

    Args:
        y_true: True binary labels.
        y_prob: Predicted positive class probabilities.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: precision, recall, pr_auc
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    pr_auc = float(auc(rec, prec))
    return prec, rec, pr_auc


def plot_confusion_matrix(
    cm: Union[np.ndarray, list],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix - Final Model",
) -> Path:
    """Generates and saves a clean, styled confusion matrix visualization."""
    cm_arr = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Benign (0)", "Predicted Malignant (1)"],
        yticklabels=["Actual Benign (0)", "Actual Malignant (1)"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("True Diagnosis", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: Union[str, Path],
    title: str = "Receiver Operating Characteristic (ROC) Curve",
) -> Path:
    """Generates and saves the ROC curve visualization."""
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        color="#2b5c8f",
        lw=2.5,
        label=f"Selected Model (AUC = {roc_auc:.4f})",
    )
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file


def plot_precision_recall_curve(
    prec: np.ndarray,
    rec: np.ndarray,
    pr_auc: float,
    save_path: Union[str, Path],
    title: str = "Precision-Recall Curve",
) -> Path:
    """Generates and saves the Precision-Recall curve visualization."""
    plt.figure(figsize=(6, 5))
    plt.plot(
        rec,
        prec,
        color="#28a745",
        lw=2.5,
        label=f"Selected Model (PR-AUC = {pr_auc:.4f})",
    )
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Union[str, Path],
    metric_col: str = "cv_roc_auc_mean",
    err_col: str = "cv_roc_auc_std",
    title: str = "Cross-Validation Model Comparison (5-Fold Stratified CV)",
) -> Path:
    """Generates and saves a horizontal bar chart comparing cross-validated models."""
    df_sorted = comparison_df.sort_values(by=metric_col, ascending=True)

    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(
        df_sorted["model"],
        df_sorted[metric_col],
        xerr=df_sorted[err_col] if err_col in df_sorted.columns else None,
        color="#3366cc",
        alpha=0.85,
        capsize=4,
        edgecolor="#1a3d7c",
    )

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width - 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="right",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    plt.xlim([0.80, 1.01])
    plt.xlabel(f"Mean {metric_col.replace('_', ' ').upper()}", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close()
    return out_file
