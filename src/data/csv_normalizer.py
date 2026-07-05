from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


TEXT_COLUMN_CANDIDATES = (
    "text",
    "full_text",
    "tweet_text",
    "content",
    "review",
    "comment",
    "message",
)
XQUIK_COLUMNS = {"tweet_id", "full_text", "conversation_id", "retweet_count"}
DEFAULT_LABEL_NAMES = ("negative", "neutral", "positive")


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _resolve_text_column(frame: pd.DataFrame, requested_column: str) -> str:
    if requested_column in frame.columns:
        return requested_column
    if requested_column != "text":
        raise ValueError(f"CSV must have a '{requested_column}' column.")

    detected = _find_column(frame, TEXT_COLUMN_CANDIDATES)
    if detected is not None:
        return detected

    available = ", ".join(str(column) for column in frame.columns)
    expected = ", ".join(TEXT_COLUMN_CANDIDATES)
    raise ValueError(
        f"CSV must include one text column. Expected one of: {expected}. "
        f"Found: {available or 'none'}."
    )


def _label_name(value: object, label_names: Sequence[str]) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in label_names:
            return normalized

    try:
        index = int(value)
    except (TypeError, ValueError):
        return "unknown"

    if index in range(len(label_names)):
        return label_names[index]
    return "unknown"


def normalize_text_dataframe(
    frame: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "label",
    label_names: Sequence[str] = DEFAULT_LABEL_NAMES,
) -> pd.DataFrame:
    text_column = _resolve_text_column(frame, text_col)
    result = frame.copy()
    result = result.rename(columns={text_column: "text"})
    if label_col in result.columns and label_col != "label":
        result = result.rename(columns={label_col: "label"})

    result["text"] = result["text"].fillna("").astype(str).str.strip()
    result = result[result["text"] != ""].reset_index(drop=True)
    if result.empty:
        raise ValueError("CSV text column is empty after blank rows are removed.")

    if "label" in result.columns:
        result["label_name"] = result["label"].map(
            lambda value: _label_name(value, label_names)
        )

    normalized_columns = {str(column).strip().lower() for column in frame.columns}
    if normalized_columns & XQUIK_COLUMNS and "source" not in result.columns:
        result["source"] = "Xquik"

    return result
