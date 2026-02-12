"""
Central configuration for Customer Feedback Intelligence.

All settings are overridable via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset
DATASET_NAME = os.getenv("DATASET_NAME", "defunct-datasets/amazon_reviews_multi")
DATASET_LANGUAGE = os.getenv("DATASET_LANGUAGE", "de")
LABEL_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}  # 5-star → 3-class (neg/neu/pos)
LABEL_NAMES = ["negative", "neutral", "positive"]
NUM_LABELS = len(LABEL_NAMES)

# Preprocessing
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "512"))
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "10"))

# Classical ML
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "50000"))
TFIDF_NGRAM_RANGE = (1, 2)

# BERT
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "deepset/gbert-base")
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "128"))
BERT_BATCH_SIZE = int(os.getenv("BERT_BATCH_SIZE", "16"))
BERT_LEARNING_RATE = float(os.getenv("BERT_LEARNING_RATE", "2e-5"))
BERT_EPOCHS = int(os.getenv("BERT_EPOCHS", "3"))
BERT_WARMUP_RATIO = float(os.getenv("BERT_WARMUP_RATIO", "0.1"))

# Azure OpenAI (LLM)
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://germanywestcentral.api.cognitive.microsoft.com/",
)
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Training
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# MLflow
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri()
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "customer-feedback-intelligence"
)

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
