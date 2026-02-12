"""
Error analysis for sentiment classification models.

Identifies patterns in misclassifications across models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.config import LABEL_NAMES

logger = logging.getLogger(__name__)


def analyze_errors(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> pd.DataFrame:
    """Build a DataFrame of misclassified examples.

    Args:
        texts: Input texts.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Name of the model for the output column.

    Returns:
        DataFrame with columns: text, true_label, pred_label, true_name, pred_name.
    """
    mask = y_true != y_pred
    errors = pd.DataFrame({
        "text": np.array(texts)[mask],
        "true_label": y_true[mask],
        "pred_label": y_pred[mask],
        "true_name": [LABEL_NAMES[i] for i in y_true[mask]],
        "pred_name": [LABEL_NAMES[i] for i in y_pred[mask]],
        "model": model_name,
    })
    return errors.reset_index(drop=True)


def error_summary(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Summarize error patterns.

    Returns:
        Dict with error counts by confusion pair, text length stats, etc.
    """
    mask = y_true != y_pred
    error_texts = np.array(texts)[mask]
    correct_texts = np.array(texts)[~mask]

    # Confusion pairs
    pairs = {}
    for t, p in zip(y_true[mask], y_pred[mask]):
        key = f"{LABEL_NAMES[t]} -> {LABEL_NAMES[p]}"
        pairs[key] = pairs.get(key, 0) + 1

    # Text length analysis
    error_lengths = [len(t) for t in error_texts] if len(error_texts) > 0 else [0]
    correct_lengths = [len(t) for t in correct_texts] if len(correct_texts) > 0 else [0]

    return {
        "total_errors": int(mask.sum()),
        "total_samples": len(y_true),
        "error_rate": float(mask.mean()),
        "confusion_pairs": dict(sorted(pairs.items(), key=lambda x: -x[1])),
        "avg_error_text_length": float(np.mean(error_lengths)),
        "avg_correct_text_length": float(np.mean(correct_lengths)),
        "shortest_error": min(error_lengths),
        "longest_error": max(error_lengths),
    }


def compare_model_errors(
    texts: list[str],
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare which samples each model gets wrong.

    Args:
        texts: Input texts.
        y_true: Ground truth labels.
        predictions: Dict of model_name -> predicted labels.

    Returns:
        DataFrame with one row per sample and a column per model (True=correct).
    """
    df = pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "true_name": [LABEL_NAMES[i] for i in y_true],
    })

    for model_name, y_pred in predictions.items():
        df[f"{model_name}_correct"] = y_true == y_pred
        df[f"{model_name}_pred"] = y_pred

    # Add count of models that got it right
    correct_cols = [c for c in df.columns if c.endswith("_correct")]
    df["n_models_correct"] = df[correct_cols].sum(axis=1)

    return df
