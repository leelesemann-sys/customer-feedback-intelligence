"""Tests for TextPreprocessor."""

import pandas as pd
import pytest

from src.data.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor(min_length=3)


class TestClean:
    def test_normal_text(self, preprocessor):
        result = preprocessor.clean("Das ist ein guter Test.")
        assert result == "Das ist ein guter Test."

    def test_removes_urls(self, preprocessor):
        result = preprocessor.clean("Schau mal https://example.com hier")
        assert "https://" not in result
        assert "example.com" not in result

    def test_removes_emails(self, preprocessor):
        result = preprocessor.clean("Kontakt: test@example.com bitte")
        assert "@" not in result

    def test_collapses_whitespace(self, preprocessor):
        result = preprocessor.clean("zu   viele    spaces")
        assert "  " not in result

    def test_preserves_german_umlauts(self, preprocessor):
        result = preprocessor.clean("Ueberraschend gute Qualitaet mit Aerger")
        assert "Ueberraschend" in result

    def test_handles_none(self, preprocessor):
        assert preprocessor.clean(None) == ""

    def test_handles_empty_string(self, preprocessor):
        assert preprocessor.clean("") == ""

    def test_handles_short_text(self, preprocessor):
        assert preprocessor.clean("ab") == ""

    def test_handles_nan(self, preprocessor):
        assert preprocessor.clean(float("nan")) == ""

    def test_max_length_truncation(self):
        p = TextPreprocessor(min_length=1, max_length=10)
        result = p.clean("This is a very long text")
        assert len(result) <= 10


class TestCleanBatch:
    def test_batch_processing(self, preprocessor):
        texts = ["Gut!", "Schlecht.", ""]
        results = preprocessor.clean_batch(texts)
        assert len(results) == 3
        assert results[2] == ""


class TestPreprocessDataframe:
    def test_adds_clean_column(self, preprocessor, sample_dataframe):
        result = preprocessor.preprocess_dataframe(sample_dataframe)
        assert "text_clean" in result.columns

    def test_drops_empty_rows(self, preprocessor):
        df = pd.DataFrame({"text": ["Good text", "", "ab"], "label": [1, 0, 0]})
        result = preprocessor.preprocess_dataframe(df)
        assert len(result) == 1
