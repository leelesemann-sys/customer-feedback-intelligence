"""
FastAPI REST API for sentiment classification.

Usage:
    uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.config import LABEL_NAMES

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Feedback Intelligence API",
    description="Sentiment classification API comparing Classical ML, BERT, and LLM models.",
    version="1.0.0",
)

# Lazy-loaded predictors
_predictors: dict = {}


def _get_predictor(model_type: str):
    """Lazy-load and cache predictors."""
    from src.inference.predictor import SentimentPredictor

    if model_type not in _predictors:
        if model_type == "classical":
            _predictors[model_type] = SentimentPredictor.from_classical()
        elif model_type == "bert":
            _predictors[model_type] = SentimentPredictor.from_bert()
        elif model_type == "llm":
            _predictors[model_type] = SentimentPredictor.from_llm()
        else:
            raise HTTPException(400, f"Unknown model type: {model_type}")

    return _predictors[model_type]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")
    model: str = Field(
        default="classical",
        description="Model to use: classical, bert, or llm",
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100)
    model: str = Field(default="classical")


class PredictionResponse(BaseModel):
    text: str
    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    model: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model: str
    count: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    return {
        "available": ["classical", "bert", "llm"],
        "labels": LABEL_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    try:
        predictor = _get_predictor(req.model)
        result = predictor.predict_single(req.text)
        return PredictionResponse(**result, model=req.model)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{req.model}' not found. Train it first.")
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(500, str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictRequest):
    try:
        predictor = _get_predictor(req.model)
        results = predictor.predict_batch(req.texts)
        predictions = [PredictionResponse(**r, model=req.model) for r in results]
        return BatchPredictionResponse(
            predictions=predictions,
            model=req.model,
            count=len(predictions),
        )
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{req.model}' not found. Train it first.")
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(500, str(e))
