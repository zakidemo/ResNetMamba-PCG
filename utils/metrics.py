"""
utils/metrics.py — Evaluation metrics for PCG classification.

Implements PhysioNet/CinC 2016 challenge metrics:
  - Sensitivity (Se) = TP / (TP + FN)
  - Specificity (Sp) = TN / (TN + FP)
  - MAcc = (Se + Sp) / 2
  - McNemar's test for statistical significance
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute PhysioNet/CinC 2016 evaluation metrics.

    Args:
        y_true: Ground truth labels (0=normal, 1=abnormal)
        y_pred: Predicted labels

    Returns:
        Dictionary with sensitivity, specificity, macc, accuracy, f1
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    macc = (sensitivity + specificity) / 2
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity /
          (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    return {
        "sensitivity": sensitivity * 100,
        "specificity": specificity * 100,
        "macc": macc * 100,
        "accuracy": accuracy * 100,
        "f1": f1 * 100,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print metrics dictionary."""
    print(f"\n{'─' * 42}")
    if prefix:
        print(f"  {prefix}")
        print(f"{'─' * 42}")
    print(f"  Sensitivity  (Se) : {metrics['sensitivity']:.2f}%")
    print(f"  Specificity  (Sp) : {metrics['specificity']:.2f}%")
    print(f"  Mean Accuracy     : {metrics['macc']:.2f}%  ← primary metric")
    print(f"  Accuracy          : {metrics['accuracy']:.2f}%")
    print(f"  F1 Score          : {metrics['f1']:.2f}%")
    print(f"  Confusion: TP={metrics['tp']} TN={metrics['tn']} "
          f"FP={metrics['fp']} FN={metrics['fn']}")
    print(f"{'─' * 42}")


def mcnemar_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    """
    McNemar's test with continuity correction.

    Args:
        errors_a: Boolean array — True where model A is wrong
        errors_b: Boolean array — True where model B is wrong

    Returns:
        Dictionary with b, c, chi2_stat, p_value, significance flags
    """
    from scipy.stats import chi2

    # Discordant pairs
    b = int(np.sum(errors_a & ~errors_b))  # A wrong, B correct
    c = int(np.sum(~errors_a & errors_b))  # A correct, B wrong

    # McNemar's chi-squared with continuity correction
    if (b + c) == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1.0 - chi2.cdf(chi2_stat, df=1)

    return {
        "b": b,
        "c": c,
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.001,
    }
