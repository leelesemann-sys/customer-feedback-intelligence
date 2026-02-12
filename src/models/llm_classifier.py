"""
LLM-based sentiment classification using Azure OpenAI.

Supports zero-shot and few-shot prompting with GPT-4o / GPT-4o-mini.
No training required — only prompt engineering and API calls.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import numpy as np
from openai import AzureOpenAI

from config.config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    LABEL_NAMES,
    NUM_LABELS,
)
from src.models.base import SentimentModel

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD) — Azure OpenAI, as of 2025
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

SYSTEM_PROMPT_ZERO_SHOT = """You are a sentiment classifier for customer reviews.
Classify the sentiment of the given text into exactly one of these categories:
- negative
- neutral
- positive

Respond with ONLY a JSON object: {"label": "<category>", "confidence": <0.0-1.0>}
No explanation, no other text."""

FEW_SHOT_EXAMPLES = [
    {"text": "Das Produkt ist ausgezeichnet! Sehr gute Qualitaet und schnelle Lieferung.", "label": "positive"},
    {"text": "Furchtbar. Kaputt angekommen und der Kundenservice war unfreundlich.", "label": "negative"},
    {"text": "Ganz okay, nichts Besonderes. Erfuellt seinen Zweck.", "label": "neutral"},
    {"text": "Absolutely love this! Best purchase I've made in years.", "label": "positive"},
    {"text": "Terrible quality. Broke after two days. Complete waste of money.", "label": "negative"},
    {"text": "It's fine. Nothing special but does what it's supposed to do.", "label": "neutral"},
]


def _build_few_shot_messages() -> list[dict[str, str]]:
    """Build few-shot example messages for the chat API."""
    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"label": ex["label"], "confidence": 0.95}),
        })
    return messages


class LLMSentimentModel(SentimentModel):
    """Sentiment classification via Azure OpenAI chat completions.

    Supports zero-shot and few-shot modes. Does not require training.

    Args:
        deployment: Azure OpenAI deployment name.
        mode: "zero_shot" or "few_shot".
        max_retries: Max retries per API call on transient errors.
        temperature: Sampling temperature (0 = deterministic).
    """

    def __init__(
        self,
        deployment: str = AZURE_OPENAI_DEPLOYMENT,
        mode: str = "zero_shot",
        max_retries: int = 3,
        temperature: float = 0.0,
    ):
        if mode not in ("zero_shot", "few_shot"):
            raise ValueError(f"Invalid mode: {mode}. Use 'zero_shot' or 'few_shot'.")

        self.deployment = deployment
        self.mode = mode
        self.max_retries = max_retries
        self.temperature = temperature
        self.name = f"llm_{deployment}_{mode}"

        self._client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self._label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        self._last_batch_results: list[dict[str, Any]] | None = None
        self._last_batch_texts: list[str] | None = None

    def _get_or_classify(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return cached results if texts match, otherwise classify."""
        if self._last_batch_texts == texts and self._last_batch_results is not None:
            return self._last_batch_results
        results = self._classify_batch(texts)
        self._last_batch_texts = texts
        self._last_batch_results = results
        return results

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict sentiment labels for a batch of texts."""
        results = self._get_or_classify(texts)
        return np.array([r["label_idx"] for r in results])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict sentiment probabilities for a batch of texts.

        Uses the model's self-reported confidence to construct a probability
        distribution. The confidence goes to the predicted class, and the
        remainder is split equally among the other classes.
        """
        results = self._get_or_classify(texts)
        probas = np.zeros((len(texts), NUM_LABELS))

        for i, r in enumerate(results):
            conf = r["confidence"]
            remaining = (1.0 - conf) / (NUM_LABELS - 1)
            probas[i, :] = remaining
            probas[i, r["label_idx"]] = conf

        return probas

    def _classify_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify a batch of texts one by one (LLM API is per-request)."""
        results = []
        for i, text in enumerate(texts):
            result = self._classify_single(text)
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(
                    "  Classified %d/%d (tokens: %d in, %d out)",
                    i + 1, len(texts),
                    self.total_input_tokens, self.total_output_tokens,
                )
        return results

    def _classify_single(self, text: str) -> dict[str, Any]:
        """Classify a single text with retry logic."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT_ZERO_SHOT}]

        if self.mode == "few_shot":
            messages.extend(_build_few_shot_messages())

        messages.append({"role": "user", "content": text})

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=50,
                )

                # Track tokens
                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens

                return self._parse_response(response.choices[0].message.content)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("API error (attempt %d): %s. Retrying in %ds...", attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("API error after %d attempts: %s", self.max_retries, e)
                    return {"label": "neutral", "label_idx": 1, "confidence": 0.33}

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse the JSON response from the LLM."""
        try:
            # Handle potential markdown code blocks
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(clean)
            label = data.get("label", "neutral").lower().strip()
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            if label not in self._label_to_idx:
                logger.warning("Unknown label '%s', defaulting to neutral", label)
                label = "neutral"

            return {
                "label": label,
                "label_idx": self._label_to_idx[label],
                "confidence": confidence,
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse LLM response: %s — content: %s", e, content[:200])
            return {"label": "neutral", "label_idx": 1, "confidence": 0.33}

    def get_cost(self) -> dict[str, float]:
        """Calculate API cost based on tracked token usage."""
        pricing = PRICING.get(self.deployment, PRICING["gpt-4o-mini"])
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]

        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4),
            "cost_per_1k_predictions": round(
                (input_cost + output_cost)
                / max(1, (self.total_input_tokens + self.total_output_tokens))
                * 1_000_000,
                4,
            ),
        }

    def reset_usage(self) -> None:
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
