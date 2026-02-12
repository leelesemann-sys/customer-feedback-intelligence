"""
Classical ML models for sentiment classification.

Implements TF-IDF vectorization + Logistic Regression, SVM, and Naive Bayes.
All models conform to the SentimentModel interface.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from config.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from src.models.base import SentimentModel

logger = logging.getLogger(__name__)

# Available classifier factories
CLASSIFIERS = {
    "logistic_regression": lambda **kw: LogisticRegression(
        C=kw.get("C", 1.0),
        max_iter=kw.get("max_iter", 1000),
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    ),
    "svm": lambda **kw: CalibratedClassifierCV(
        LinearSVC(
            C=kw.get("C", 1.0),
            class_weight="balanced",
            max_iter=kw.get("max_iter", 5000),
            random_state=42,
        ),
        cv=3,
    ),
    "naive_bayes": lambda **kw: MultinomialNB(
        alpha=kw.get("alpha", 1.0),
    ),
}


class ClassicalSentimentModel(SentimentModel):
    """TF-IDF + Classical Classifier pipeline.

    Wraps sklearn Pipeline with TfidfVectorizer and a classifier.
    Supports Logistic Regression, SVM, and Naive Bayes.
    """

    def __init__(
        self,
        classifier_name: str = "logistic_regression",
        tfidf_max_features: int = TFIDF_MAX_FEATURES,
        tfidf_ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
        **classifier_kwargs,
    ):
        if classifier_name not in CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier: {classifier_name}. "
                f"Choose from: {list(CLASSIFIERS.keys())}"
            )

        self.name = f"classical_{classifier_name}"
        self.classifier_name = classifier_name

        vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

        classifier = CLASSIFIERS[classifier_name](**classifier_kwargs)

        self.pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", classifier),
        ])

        self._is_trained = False

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: Optional[list[str]] = None,
        val_labels: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Train the pipeline on text data.

        Returns:
            Dict with training duration and optional validation score.
        """
        logger.info("Training %s on %d samples...", self.name, len(train_texts))
        start = time.perf_counter()

        self.pipeline.fit(train_texts, train_labels)
        self._is_trained = True

        duration = time.perf_counter() - start
        result = {"train_duration_s": round(duration, 2)}

        if val_texts is not None and val_labels is not None:
            val_pred = self.pipeline.predict(val_texts)
            from sklearn.metrics import f1_score
            result["val_f1_weighted"] = float(
                f1_score(val_labels, val_pred, average="weighted")
            )

        logger.info("Training complete in %.2fs", duration)
        return result

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict class labels."""
        if not self._is_trained:
            raise RuntimeError(f"{self.name} has not been trained yet.")
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_trained:
            raise RuntimeError(f"{self.name} has not been trained yet.")
        return self.pipeline.predict_proba(texts)

    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Saved %s to %s", self.name, path)

    @classmethod
    def load(cls, path: str, classifier_name: str = "logistic_regression") -> "ClassicalSentimentModel":
        """Load a saved pipeline from disk."""
        model = cls.__new__(cls)
        model.pipeline = joblib.load(path)
        model.classifier_name = classifier_name
        model.name = f"classical_{classifier_name}"
        model._is_trained = True
        logger.info("Loaded %s from %s", model.name, path)
        return model
