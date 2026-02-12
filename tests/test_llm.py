"""Tests for LLM sentiment classifier."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.llm_classifier import (
    LLMSentimentModel,
    _build_few_shot_messages,
)


class TestLLMSentimentModel:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            LLMSentimentModel(mode="invalid")

    def test_name_includes_deployment_and_mode(self):
        with patch("src.models.llm_classifier.AzureOpenAI"):
            model = LLMSentimentModel(deployment="gpt-4o-mini", mode="zero_shot")
            assert model.name == "llm_gpt-4o-mini_zero_shot"

    def test_few_shot_messages_structure(self):
        messages = _build_few_shot_messages()
        assert len(messages) == 12  # 6 examples × 2 messages each
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        # Each assistant message should be valid JSON
        for m in messages:
            if m["role"] == "assistant":
                data = json.loads(m["content"])
                assert "label" in data
                assert "confidence" in data

    @patch("src.models.llm_classifier.AzureOpenAI")
    def test_predict_parses_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock API response
        mock_choice = MagicMock()
        mock_choice.message.content = '{"label": "positive", "confidence": 0.92}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        model = LLMSentimentModel(deployment="gpt-4o-mini", mode="zero_shot")
        preds = model.predict(["Great product!"])

        assert len(preds) == 1
        assert preds[0] == 2  # positive = index 2

    @patch("src.models.llm_classifier.AzureOpenAI")
    def test_predict_proba_shape(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '{"label": "negative", "confidence": 0.85}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        model = LLMSentimentModel(deployment="gpt-4o-mini", mode="zero_shot")
        proba = model.predict_proba(["Terrible!"])

        assert proba.shape == (1, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert proba[0, 0] == 0.85  # negative class has highest prob

    @patch("src.models.llm_classifier.AzureOpenAI")
    def test_handles_invalid_json_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "I think this is positive."  # Not JSON
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        model = LLMSentimentModel(deployment="gpt-4o-mini", mode="zero_shot")
        preds = model.predict(["Some text"])

        # Should default to neutral (1) on parse failure
        assert preds[0] == 1

    @patch("src.models.llm_classifier.AzureOpenAI")
    def test_handles_unknown_label(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '{"label": "sehr_positiv", "confidence": 0.9}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        model = LLMSentimentModel(deployment="gpt-4o-mini", mode="zero_shot")
        preds = model.predict(["Some text"])

        assert preds[0] == 1  # defaults to neutral

    @patch("src.models.llm_classifier.AzureOpenAI")
    def test_cost_tracking(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '{"label": "positive", "confidence": 0.9}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response

        model = LLMSentimentModel(deployment="gpt-4o-mini", mode="zero_shot")
        model.predict(["text1", "text2"])

        cost = model.get_cost()
        assert cost["input_tokens"] == 200  # 100 × 2
        assert cost["output_tokens"] == 40  # 20 × 2
        assert cost["total_cost_usd"] > 0

    @patch("src.models.llm_classifier.AzureOpenAI")
    def test_reset_usage(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        model = LLMSentimentModel(deployment="gpt-4o-mini", mode="zero_shot")
        model.total_input_tokens = 1000
        model.total_output_tokens = 200
        model.reset_usage()

        assert model.total_input_tokens == 0
        assert model.total_output_tokens == 0
