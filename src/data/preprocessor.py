"""
Text preprocessing pipeline.

Ensures identical preprocessing in training and inference.
Designed for German text but works with any language.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd


class TextPreprocessor:
    """Stateless text preprocessor for consistency between training and inference.

    Applies minimal cleaning to preserve semantic content for transformer models.
    For classical ML, additional steps (lowercasing, stopword removal) are applied
    at the TF-IDF vectorizer level.
    """

    def __init__(self, min_length: int = 3, max_length: Optional[int] = None):
        self.min_length = min_length
        self.max_length = max_length

    def clean(self, text: str) -> str:
        """Clean a single text string.

        Steps:
        1. Handle missing/empty input
        2. Normalize unicode (umlauts, accents)
        3. Remove URLs
        4. Remove email addresses
        5. Collapse whitespace
        6. Truncate if max_length set
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        text = text.strip()
        if len(text) < self.min_length:
            return ""

        # Normalize unicode (keep German umlauts intact via NFC)
        text = unicodedata.normalize("NFC", text)

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Replace multiple whitespace/newlines with single space
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate
        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length]

        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a list of texts."""
        return [self.clean(t) for t in texts]

    def preprocess_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Clean text column in a DataFrame in-place.

        Adds 'text_clean' column and drops rows where cleaning produces empty string.
        """
        df = df.copy()
        df["text_clean"] = df[text_col].apply(self.clean)
        df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
        return df
