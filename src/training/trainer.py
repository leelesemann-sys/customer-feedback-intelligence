"""
Unified training orchestration.

Handles training for all model types with MLflow experiment tracking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from config.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RESULTS_DIR,
)
from src.evaluation.metrics import compute_metrics, measure_latency
from src.models.base import SentimentModel

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_and_evaluate(
    model: SentimentModel,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    log_to_mlflow: bool = True,
    save_model: bool = True,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a model, evaluate on val+test, and log everything.

    Args:
        model: Any SentimentModel instance.
        train_texts, train_labels: Training data.
        val_texts, val_labels: Validation data.
        test_texts, test_labels: Test data.
        log_to_mlflow: Whether to log to MLflow.
        save_model: Whether to save model artifacts.
        extra_params: Additional params to log (e.g., hyperparameters).

    Returns:
        Dict with train_result, val_metrics, test_metrics, latency.
    """
    run_context = mlflow.start_run(run_name=model.name) if log_to_mlflow else _null_context()

    with run_context:
        if log_to_mlflow and extra_params:
            mlflow.log_params(extra_params)

        # Train
        logger.info("Training %s...", model.name)
        train_result = model.train(train_texts, train_labels, val_texts, val_labels)

        if log_to_mlflow:
            mlflow.log_metrics({
                f"train_{k}": v for k, v in train_result.items() if isinstance(v, (int, float))
            })

        # Evaluate on validation set
        val_pred = model.predict(val_texts)
        val_proba = model.predict_proba(val_texts)
        val_metrics = compute_metrics(np.array(val_labels), val_pred, val_proba)

        # Evaluate on test set
        test_pred = model.predict(test_texts)
        test_proba = model.predict_proba(test_texts)
        test_metrics = compute_metrics(np.array(test_labels), test_pred, test_proba)

        # Latency
        sample = test_texts[:100]
        latency = measure_latency(model, sample)

        if log_to_mlflow:
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
            mlflow.log_metrics({f"latency_{k}": v for k, v in latency.items()})

        # Save
        if save_model:
            model_path = str(MODELS_DIR / f"{model.name}.joblib")
            try:
                model.save(model_path)
            except NotImplementedError:
                logger.info("Model %s does not support saving.", model.name)

        # Save metrics to JSON
        all_results = {
            "model": model.name,
            "train": train_result,
            "val": {k: v for k, v in val_metrics.items() if k != "classification_report"},
            "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
            "latency": latency,
            "test_classification_report": test_metrics.get("classification_report"),
        }

        metrics_path = RESULTS_DIR / "metrics" / f"{model.name}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(
            "%s — Test F1: %.4f | Accuracy: %.4f | Latency: %.2f ms/sample",
            model.name,
            test_metrics["f1_weighted"],
            test_metrics["accuracy"],
            latency["per_sample_ms"],
        )

    return all_results


class _null_context:
    """No-op context manager when MLflow is disabled."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
