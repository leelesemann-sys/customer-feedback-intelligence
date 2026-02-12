"""
Optuna-based hyperparameter optimization for classical ML models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score

from src.models.classical import ClassicalSentimentModel

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_classical(
    classifier_name: str,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    n_trials: int = 50,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search for a classical model.

    Args:
        classifier_name: One of 'logistic_regression', 'svm', 'naive_bayes'.
        train_texts, train_labels: Training data.
        val_texts, val_labels: Validation data.
        n_trials: Number of Optuna trials.

    Returns:
        Dict with best_params, best_f1, best_model.
    """

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, classifier_name)

        model = ClassicalSentimentModel(
            classifier_name=classifier_name,
            tfidf_max_features=trial.suggest_categorical(
                "tfidf_max_features", [10000, 30000, 50000]
            ),
            **params,
        )
        model.train(train_texts, train_labels)
        pred = model.predict(val_texts)
        return float(f1_score(val_labels, pred, average="weighted"))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(
        "Best %s: F1=%.4f, params=%s",
        classifier_name, best.value, best.params,
    )

    # Retrain with best params
    best_params = {k: v for k, v in best.params.items() if k != "tfidf_max_features"}
    best_model = ClassicalSentimentModel(
        classifier_name=classifier_name,
        tfidf_max_features=best.params["tfidf_max_features"],
        **best_params,
    )
    best_model.train(train_texts, train_labels)

    return {
        "best_params": best.params,
        "best_f1": best.value,
        "best_model": best_model,
        "n_trials": n_trials,
    }


def _suggest_params(trial: optuna.Trial, classifier_name: str) -> dict:
    """Suggest hyperparameters based on classifier type."""
    if classifier_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        }
    elif classifier_name == "svm":
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        }
    elif classifier_name == "naive_bayes":
        return {
            "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
        }
    else:
        return {}
