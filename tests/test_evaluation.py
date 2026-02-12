"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


class TestComputeMetrics:
    def test_returns_all_keys(self, sample_predictions):
        y_true, y_pred, y_proba = sample_predictions
        results = compute_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in results
        assert "f1_weighted" in results
        assert "f1_macro" in results
        assert "precision_weighted" in results
        assert "recall_weighted" in results
        assert "confusion_matrix" in results
        assert "classification_report" in results
        assert "roc_auc_weighted" in results

    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        results = compute_metrics(y, y)
        assert results["accuracy"] == 1.0
        assert results["f1_weighted"] == 1.0

    def test_metrics_in_valid_range(self, sample_predictions):
        y_true, y_pred, y_proba = sample_predictions
        results = compute_metrics(y_true, y_pred, y_proba)

        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["f1_weighted"] <= 1.0
        assert 0.0 <= results["precision_weighted"] <= 1.0
        assert 0.0 <= results["recall_weighted"] <= 1.0

    def test_confusion_matrix_shape(self, sample_predictions):
        y_true, y_pred, _ = sample_predictions
        results = compute_metrics(y_true, y_pred)
        cm = results["confusion_matrix"]
        assert len(cm) == 3
        assert len(cm[0]) == 3

    def test_without_proba(self, sample_predictions):
        y_true, y_pred, _ = sample_predictions
        results = compute_metrics(y_true, y_pred)
        assert "roc_auc_weighted" not in results
