"""
Unit tests for the Breast Cancer Evaluation module.

Validates metric computation correctness, sensitivity and specificity mathematical logic,
confusion matrix dimension enforcement, and curve data generation.
"""

import numpy as np
import pytest

from member3b.breast_cancer.src.evaluate import (
    calculate_classification_metrics,
    compute_roc_curve_data,
    compute_pr_curve_data,
)


def test_calculate_classification_metrics_perfect_prediction():
    """Verify metrics calculation for a hypothetical perfect prediction."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["tn"] == 2
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_calculate_classification_metrics_math_correctness():
    """
    Verify mathematical calculation of sensitivity, specificity, precision, and recall.
    Matrix:
        TN = 70, FP = 10 -> Actual Negative (0) = 80
        FN = 5,  TP = 15 -> Actual Positive (1) = 20
    """
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.array([0] * 70 + [1] * 10 + [0] * 5 + [1] * 15)

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics["tn"] == 70
    assert metrics["fp"] == 10
    assert metrics["fn"] == 5
    assert metrics["tp"] == 15

    expected_sensitivity = 15 / (15 + 5)  # 0.75
    expected_specificity = 70 / (70 + 10)  # 0.875
    expected_precision = 15 / (15 + 10)  # 0.60
    expected_accuracy = (70 + 15) / 100  # 0.85

    assert abs(metrics["sensitivity"] - expected_sensitivity) < 1e-6
    assert abs(metrics["recall"] - expected_sensitivity) < 1e-6
    assert abs(metrics["specificity"] - expected_specificity) < 1e-6
    assert abs(metrics["precision"] - expected_precision) < 1e-6
    assert abs(metrics["accuracy"] - expected_accuracy) < 1e-6


def test_roc_and_pr_curve_data_ranges():
    """Verify that ROC and PR curve generators produce values strictly in [0, 1]."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.3, 0.8, 0.4, 0.7])

    fpr, tpr, roc_auc = compute_roc_curve_data(y_true, y_prob)
    assert 0.0 <= roc_auc <= 1.0
    assert np.all(fpr >= 0.0) and np.all(fpr <= 1.0)
    assert np.all(tpr >= 0.0) and np.all(tpr <= 1.0)

    prec, rec, pr_auc = compute_pr_curve_data(y_true, y_prob)
    assert 0.0 <= pr_auc <= 1.0
    assert np.all(prec >= 0.0) and np.all(prec <= 1.0)
    assert np.all(rec >= 0.0) and np.all(rec <= 1.0)
