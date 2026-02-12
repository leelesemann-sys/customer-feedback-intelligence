"""
Production inference module.

Provides a unified interface for sentiment prediction using any trained model.
Supports single and batch predictions with optional preprocessing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config.config import LABEL_NAMES, MODELS_DIR
from src.data.preprocessor import TextPreprocessor
from src.models.base import SentimentModel

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """High-level predictor that wraps any SentimentModel.

    Handles preprocessing, prediction, and result formatting.

    Args:
        model: A trained SentimentModel instance.
        preprocess: Whether to apply text preprocessing before prediction.
    """

    def __init__(self, model: SentimentModel, preprocess: bool = True):
        self.model = model
        self.preprocessor = TextPreprocessor() if preprocess else None

    def predict_single(self, text: str) -> dict[str, Any]:
        """Predict sentiment for a single text.

        Returns:
            Dict with label, label_name, confidence, and probabilities.
        """
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict sentiment for a batch of texts.

        Returns:
            List of dicts, each with label, label_name, confidence, probabilities.
        """
        if self.preprocessor:
            texts = self.preprocessor.clean_batch(texts)

        labels = self.model.predict(texts)
        probas = self.model.predict_proba(texts)

        results = []
        for i, (label, proba) in enumerate(zip(labels, probas)):
            results.append({
                "text": texts[i],
                "label": int(label),
                "label_name": LABEL_NAMES[int(label)],
                "confidence": float(proba.max()),
                "probabilities": {
                    name: float(p)
                    for name, p in zip(LABEL_NAMES, proba)
                },
            })

        return results

    @classmethod
    def from_classical(cls, model_name: str = "classical_logistic_regression") -> "SentimentPredictor":
        """Load a classical ML model from disk."""
        from src.models.classical import ClassicalSentimentModel

        model_path = str(MODELS_DIR / f"{model_name}.joblib")
        model = ClassicalSentimentModel.load(model_path)
        return cls(model=model)

    @classmethod
    def from_bert(cls, model_path: str | None = None) -> "SentimentPredictor":
        """Load a fine-tuned BERT model from disk."""
        from src.models.bert_classifier import BertSentimentModel

        if model_path is None:
            model_path = str(MODELS_DIR / "bert_german_sentiment")
        model = BertSentimentModel.load(model_path)
        return cls(model=model, preprocess=False)  # BERT tokenizer handles preprocessing

    @classmethod
    def from_llm(
        cls,
        deployment: str = "gpt-4o-mini",
        mode: str = "zero_shot",
    ) -> "SentimentPredictor":
        """Create an LLM-based predictor (no model loading needed)."""
        from src.models.llm_classifier import LLMSentimentModel

        model = LLMSentimentModel(deployment=deployment, mode=mode)
        return cls(model=model)
