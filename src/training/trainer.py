"""
Unified training orchestration.

Handles training for all model types with MLflow experiment tracking.
"""

from __future__ import annotations

import json
import logging
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
    dataset_name: str = "german",
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
        dataset_name: Dataset identifier for metrics filename (e.g., 'german', 'yelp').
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

        # Include dataset name in filename to avoid cross-dataset overwrites
        suffix = f"_{dataset_name}" if dataset_name != "german" else ""
        metrics_path = RESULTS_DIR / "metrics" / f"{model.name}{suffix}.json"
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


def train_multi_seed(
    model_factory,
    dataset_loader,
    seeds: list[int],
    dataset_name: str = "german",
    max_train_samples: int | None = None,
) -> dict[str, Any]:
    """Train a model across multiple seeds and aggregate metrics.

    Args:
        model_factory: Callable that returns a fresh SentimentModel instance.
        dataset_loader: Callable(seed) → dict with 'train', 'val', 'test' splits.
        seeds: List of random seeds to use.
        dataset_name: Name for logging.
        max_train_samples: Cap on training samples.

    Returns:
        Dict with per-seed results, mean, and std for key metrics.
    """
    from src.data.preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    all_runs = []

    for seed in seeds:
        logger.info("=" * 50)
        logger.info("Seed %d / %d: %d", seeds.index(seed) + 1, len(seeds), seed)
        logger.info("=" * 50)

        splits = dataset_loader(seed=seed)
        for split_name in splits:
            splits[split_name] = preprocessor.preprocess_dataframe(splits[split_name])

        model = model_factory()
        result = train_and_evaluate(
            model=model,
            train_texts=splits["train"]["text_clean"].tolist(),
            train_labels=splits["train"]["label"].tolist(),
            val_texts=splits["val"]["text_clean"].tolist(),
            val_labels=splits["val"]["label"].tolist(),
            test_texts=splits["test"]["text_clean"].tolist(),
            test_labels=splits["test"]["label"].tolist(),
            log_to_mlflow=False,
            save_model=False,
            extra_params={"seed": seed, "dataset": dataset_name},
        )
        all_runs.append(result)

    # Aggregate metrics across seeds
    key_metrics = ["accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted"]
    aggregated = {}
    for metric in key_metrics:
        values = [r["test"][metric] for r in all_runs if metric in r.get("test", {})]
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    logger.info("=" * 60)
    logger.info("MULTI-SEED RESULTS (%d seeds)", len(seeds))
    logger.info("=" * 60)
    for metric, stats in aggregated.items():
        logger.info(
            "  %s: %.4f +/- %.4f",
            metric, stats["mean"], stats["std"],
        )

    return {
        "seeds": seeds,
        "runs": all_runs,
        "aggregated": aggregated,
    }


def compute_learning_curve(
    model_factory,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    train_sizes: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Compute learning curve by training on increasing data sizes.

    Args:
        model_factory: Callable that returns a fresh SentimentModel instance.
        train_texts, train_labels: Full training data.
        test_texts, test_labels: Fixed test data.
        train_sizes: List of training set sizes to evaluate.

    Returns:
        List of dicts with train_size, f1_weighted, accuracy.
    """
    if train_sizes is None:
        train_sizes = [500, 1000, 2000, 5000, 10000]

    results = []
    for size in train_sizes:
        if size > len(train_texts):
            continue

        # Stratified subsample
        from sklearn.model_selection import train_test_split
        sub_texts, _, sub_labels, _ = train_test_split(
            train_texts, train_labels, train_size=size,
            random_state=42, stratify=train_labels,
        )

        model = model_factory()
        model.train(sub_texts, sub_labels)
        preds = model.predict(test_texts)

        from sklearn.metrics import f1_score, accuracy_score
        f1 = float(f1_score(test_labels, preds, average="weighted"))
        acc = float(accuracy_score(test_labels, preds))

        results.append({"train_size": size, "f1_weighted": f1, "accuracy": acc})
        logger.info("  n=%5d → F1=%.4f, Acc=%.4f", size, f1, acc)

    return results


class _null_context:
    """No-op context manager when MLflow is disabled."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
