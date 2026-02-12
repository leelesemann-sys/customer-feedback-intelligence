"""
Dataset loading utilities.

Supports multiple datasets:
- Yelp Reviews (650K English, 5-star → 3-class)
- German Sentiment (8.7K German, 3-class)
- Local CSV files
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config.config import LABEL_NAMES, RANDOM_SEED

logger = logging.getLogger(__name__)

# Star-to-sentiment mapping for 5-class → 3-class
STAR_TO_SENTIMENT = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}  # Yelp: 0-4 labels


def load_yelp_reviews(
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Load Yelp Reviews from HuggingFace (650K train, 50K test).

    Maps 5-star ratings to 3 classes: negative (1-2), neutral (3), positive (4-5).

    Args:
        max_train_samples: Cap training set size for faster iteration.
        max_test_samples: Cap test set size (stratified). Ensures all models
            are evaluated on the exact same test subset when using the same seed.
        seed: Random seed for sampling. Defaults to RANDOM_SEED (42).

    Returns:
        Dict with 'train', 'val', 'test' DataFrames.
    """
    logger.info("Loading Yelp Reviews...")

    ds = load_dataset("Yelp/yelp_review_full")

    # Process train split → train + val
    train_df = ds["train"].to_pandas()
    train_df = _standardize_yelp(train_df)

    if max_train_samples:
        train_df = train_df.sample(
            n=min(max_train_samples, len(train_df)), random_state=seed
        ).reset_index(drop=True)

    # Split train into train+val (85/15)
    train_split, val_split = train_test_split(
        train_df, test_size=0.15, random_state=seed, stratify=train_df["label"]
    )

    # Process test split
    test_df = ds["test"].to_pandas()
    test_df = _standardize_yelp(test_df)

    # Subsample test set (stratified, fixed seed) for consistent cross-model eval
    if max_test_samples and max_test_samples < len(test_df):
        test_df, _ = train_test_split(
            test_df, train_size=max_test_samples, random_state=RANDOM_SEED,
            stratify=test_df["label"]
        )
        test_df = test_df.reset_index(drop=True)

    splits = {
        "train": train_split.reset_index(drop=True),
        "val": val_split.reset_index(drop=True),
        "test": test_df,
    }

    for name, df in splits.items():
        logger.info("  %s: %d samples", name, len(df))

    return splits


def load_german_sentiment(
    max_train_samples: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Load German Sentiment dataset from HuggingFace (8.7K total).

    Already has train/val/test splits with 3-class labels.
    Test set is fixed (1490 samples) — no subsampling needed since it's small.

    Args:
        max_train_samples: Cap training set size for faster iteration.
        seed: Random seed for train subsampling. Defaults to RANDOM_SEED (42).

    Returns:
        Dict with 'train', 'val', 'test' DataFrames.
    """
    logger.info("Loading German Sentiment dataset...")

    ds = load_dataset("sepidmnorozy/German_sentiment")

    splits = {}
    for split_name, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        df = ds[hf_split].to_pandas()
        df = df.rename(columns={"text": "text", "label": "label"})
        df["label_name"] = df["label"].map(lambda x: LABEL_NAMES[x] if x in range(len(LABEL_NAMES)) else "unknown")

        # Drop empty
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

        if max_train_samples and split_name == "train":
            df = df.sample(
                n=min(max_train_samples, len(df)), random_state=seed
            ).reset_index(drop=True)

        splits[split_name] = df
        logger.info("  %s: %d samples", split_name, len(df))

    return splits


def load_from_csv(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load a local CSV file for inference or additional evaluation."""
    df = pd.read_csv(path)
    df = df.rename(columns={text_col: "text"})

    if label_col in df.columns and label_col != "label":
        df = df.rename(columns={label_col: "label"})

    if "label" in df.columns:
        df["label_name"] = df["label"].map(
            lambda x: LABEL_NAMES[x] if x in range(len(LABEL_NAMES)) else "unknown"
        )

    return df


def get_label_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Return label counts for a DataFrame."""
    return df["label"].value_counts().sort_index().to_dict()


def _standardize_yelp(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Yelp 5-class to 3-class and standardize columns."""
    df = df.copy()
    df["label"] = df["label"].map(STAR_TO_SENTIMENT)
    df["label_name"] = df["label"].map(lambda x: LABEL_NAMES[x])
    df = df[["text", "label", "label_name"]].copy()
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    return df
