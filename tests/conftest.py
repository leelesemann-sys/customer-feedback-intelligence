"""Shared test fixtures."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_texts():
    """German review texts for testing."""
    return [
        "Das Produkt ist ausgezeichnet! Sehr gute Qualitaet.",
        "Furchtbar. Kaputt angekommen, nie wieder.",
        "Ganz okay, nichts Besonderes.",
        "Schnelle Lieferung und tolles Design!",
        "Mangelhaft. Funktioniert nicht wie beschrieben.",
    ]


@pytest.fixture
def sample_labels():
    """Labels matching sample_texts: pos, neg, neu, pos, neg."""
    return [2, 0, 1, 2, 0]


@pytest.fixture
def sample_dataframe(sample_texts, sample_labels):
    """DataFrame with text and labels."""
    return pd.DataFrame({
        "text": sample_texts,
        "label": sample_labels,
    })


@pytest.fixture
def sample_predictions():
    """Dummy predictions for metrics testing."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2])
    y_pred = np.array([0, 1, 2, 0, 0, 2, 1, 2])
    y_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.4, 0.3, 0.3],
        [0.1, 0.2, 0.7],
        [0.3, 0.4, 0.3],
        [0.1, 0.1, 0.8],
    ])
    return y_true, y_pred, y_proba
