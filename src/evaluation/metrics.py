"""
Evaluation metrics for sentiment classification.

Computes standard classification metrics and formats them for comparison.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.config import LABEL_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (n_samples, n_classes). Optional.

    Returns:
        Dict with accuracy, f1_weighted, f1_macro, precision, recall,
        confusion_matrix, classification_report, and optionally roc_auc.
    """
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=LABEL_NAMES, output_dict=True
        ),
    }

    if y_proba is not None:
        try:
            results["roc_auc_weighted"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            results["roc_auc_weighted"] = None

    return results


def measure_latency(
    model,
    texts: list[str],
    n_runs: int = 3,
) -> dict[str, float]:
    """Measure average inference latency.

    Args:
        model: A SentimentModel instance.
        texts: Sample texts to predict.
        n_runs: Number of runs to average.

    Returns:
        Dict with total_ms, per_sample_ms, samples_per_second.
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(texts)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_total = np.mean(times)
    n_samples = len(texts)

    return {
        "total_ms": round(avg_total * 1000, 2),
        "per_sample_ms": round((avg_total / n_samples) * 1000, 2),
        "samples_per_second": round(n_samples / avg_total, 1),
    }
