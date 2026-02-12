"""Shared test fixtures."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_texts():
    """German review texts for testing (3+ per class for CV compatibility)."""
    return [
        "Das Produkt ist ausgezeichnet! Sehr gute Qualitaet.",
        "Schnelle Lieferung und tolles Design!",
        "Hervorragend, kann ich nur empfehlen.",
        "Furchtbar. Kaputt angekommen, nie wieder.",
        "Mangelhaft. Funktioniert nicht wie beschrieben.",
        "Schlecht verarbeitet, sehr enttaeuschend.",
        "Ganz okay, nichts Besonderes.",
        "Durchschnittlich, weder gut noch schlecht.",
        "Normal, passt schon so.",
    ]


@pytest.fixture
def sample_labels():
    """Labels matching sample_texts: 3x pos, 3x neg, 3x neu."""
    return [2, 2, 2, 0, 0, 0, 1, 1, 1]


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
