"""
Abstract base class for all sentiment models.

Defines a unified interface so that Classical ML, BERT, and LLM models
can be trained, evaluated, and compared with the same code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class SentimentModel(ABC):
    """Base interface for sentiment classification models.

    Every model must implement predict() and predict_proba().
    train() is optional (LLM models don't train).
    """

    name: str = "base"

    @abstractmethod
    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict class labels for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            Array of integer labels, shape (n_samples,).
        """

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            Array of probabilities, shape (n_samples, n_classes).
        """

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: Optional[list[str]] = None,
        val_labels: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Train the model. Optional — LLM models skip this.

        Returns:
            Dict with training metrics (loss, duration, etc.).
        """
        raise NotImplementedError(f"{self.name} does not support training.")

    def save(self, path: str) -> None:
        """Save model artifacts to disk."""
        raise NotImplementedError(f"{self.name} does not support saving.")

    @classmethod
    def load(cls, path: str) -> "SentimentModel":
        """Load model artifacts from disk."""
        raise NotImplementedError(f"{cls.name} does not support loading.")
