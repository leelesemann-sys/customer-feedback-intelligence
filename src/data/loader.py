"""
Dataset loading utilities.

Loads and prepares datasets from HuggingFace, Kaggle, or local files.
Converts star ratings to 3-class sentiment labels (negative/neutral/positive).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset

from config.config import (
    DATA_PROCESSED,
    DATASET_LANGUAGE,
    DATASET_NAME,
    LABEL_MAP,
    LABEL_NAMES,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


def load_amazon_reviews(
    language: str = DATASET_LANGUAGE,
    max_train_samples: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Load Amazon Reviews Multilingual from HuggingFace.

    Args:
        language: Language split to load (default: 'de').
        max_train_samples: Cap training set size for faster iteration.

    Returns:
        Dict with 'train', 'val', 'test' DataFrames containing
        columns ['text', 'label', 'label_name'].
    """
    logger.info("Loading Amazon Reviews Multilingual (lang=%s)...", language)

    ds = load_dataset(DATASET_NAME, language)

    splits = {}
    for split_name, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        df = ds[hf_split].to_pandas()

        # Map 1-5 stars to 3-class
        df["label"] = df["stars"].map(LABEL_MAP)
        df["label_name"] = df["label"].map(lambda x: LABEL_NAMES[x])
        df = df.rename(columns={"review_body": "text"})
        df = df[["text", "label", "label_name"]].copy()

        # Drop empty texts
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

        if max_train_samples and split_name == "train":
            df = df.sample(n=min(max_train_samples, len(df)), random_state=RANDOM_SEED)
            df = df.reset_index(drop=True)

        splits[split_name] = df
        logger.info("  %s: %d samples", split_name, len(df))

    return splits


def load_from_csv(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load a local CSV file for inference or additional evaluation.

    Args:
        path: Path to CSV file.
        text_col: Column name containing review text.
        label_col: Column name containing labels (optional).

    Returns:
        DataFrame with standardized columns.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={text_col: "text"})

    if label_col in df.columns and label_col != "label":
        df = df.rename(columns={label_col: "label"})

    if "label" in df.columns:
        df["label_name"] = df["label"].map(lambda x: LABEL_NAMES[x] if x in range(len(LABEL_NAMES)) else "unknown")

    return df


def get_label_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Return label counts for a DataFrame."""
    return df["label"].value_counts().sort_index().to_dict()
