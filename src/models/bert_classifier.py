"""
Fine-tuned BERT model for sentiment classification.

Wraps HuggingFace transformers with the SentimentModel interface.
Supports gbert-base, german-sentiment-bert, and any AutoModel.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score
import datasets as hf_datasets

from config.config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    BERT_WARMUP_RATIO,
    MODELS_DIR,
    NUM_LABELS,
)
from src.data.dataset import SentimentDataset
from src.models.base import SentimentModel

logger = logging.getLogger(__name__)


def _compute_hf_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }


class BertSentimentModel(SentimentModel):
    """Fine-tuned BERT for sentiment classification.

    Uses HuggingFace Trainer API for training with early stopping,
    fp16, and learning rate scheduling.
    """

    name = "bert_finetuned"

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        max_length: int = BERT_MAX_LENGTH,
        batch_size: int = BERT_BATCH_SIZE,
        learning_rate: float = BERT_LEARNING_RATE,
        epochs: int = BERT_EPOCHS,
        warmup_ratio: float = BERT_WARMUP_RATIO,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self._is_trained = False

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: Optional[list[str]] = None,
        val_labels: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Fine-tune BERT on training data.

        Uses HuggingFace Trainer with early stopping on validation F1.
        """
        logger.info(
            "Fine-tuning %s on %d samples (device=%s)...",
            self.model_name, len(train_texts), self.device,
        )
        start = time.perf_counter()

        # Initialize fresh model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=NUM_LABELS
        )

        # Create datasets
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )

        eval_dataset = None
        if val_texts and val_labels:
            eval_dataset = SentimentDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )

        # Output directory
        output_dir = str(MODELS_DIR / "bert_checkpoints")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1_weighted" if eval_dataset else None,
            greater_is_better=True,
            logging_steps=50,
            save_total_limit=2,
            seed=42,
            report_to="none",  # We use our own MLflow logging
        )

        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=_compute_hf_metrics,
            callbacks=callbacks,
        )

        # Train
        train_result = trainer.train()
        self._is_trained = True

        duration = time.perf_counter() - start

        result = {
            "train_duration_s": round(duration, 2),
            "train_loss": round(train_result.training_loss, 4),
            "train_samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0), 1),
        }

        if eval_dataset:
            eval_metrics = trainer.evaluate()
            result["val_f1_weighted"] = eval_metrics.get("eval_f1_weighted", 0)
            result["val_accuracy"] = eval_metrics.get("eval_accuracy", 0)

        logger.info("Fine-tuning complete in %.1fs", duration)
        return result

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities in batches."""
        if not self._is_trained and self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")

        self.model.eval()
        self.model.to(self.device)

        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def save(self, path: str) -> None:
        """Save model and tokenizer."""
        save_dir = Path(path).with_suffix("")
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        logger.info("Saved BERT model to %s", save_dir)

    @classmethod
    def load(cls, path: str) -> "BertSentimentModel":
        """Load model and tokenizer from directory."""
        instance = cls.__new__(cls)
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.model.to(instance.device)
        instance.max_length = 128
        instance.batch_size = 32
        instance._is_trained = True
        instance.name = "bert_finetuned"
        logger.info("Loaded BERT model from %s (device=%s)", path, instance.device)
        return instance
