"""
PyTorch Dataset for BERT-based models.

Handles tokenization and batching for transformer training/inference.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SentimentDataset(Dataset):
    """PyTorch Dataset that tokenizes text on-the-fly.

    Args:
        texts: List of input text strings.
        labels: Optional list of integer labels.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: Optional[list[int]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
