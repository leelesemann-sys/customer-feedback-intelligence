# Customer Feedback Intelligence

A systematic comparison of three ML approaches for German sentiment classification: **Classical ML** (TF-IDF + traditional classifiers), **Fine-tuned BERT** (deepset/gbert-base), and **LLM-based** (Azure OpenAI GPT-4o zero/few-shot).

Trained and evaluated on the [German Sentiment Dataset](https://huggingface.co/datasets/sepidmnorozy/German_sentiment) (8.7K samples, 3-class: negative/neutral/positive).

## Results

| Model | F1 (weighted) | Accuracy | Latency (ms/sample) | Cost/1K ($) |
|-------|:---:|:---:|:---:|:---:|
| **SVM (TF-IDF)** | **0.8562** | **0.8725** | 0.1 | 0 |
| Logistic Regression (TF-IDF) | 0.8505 | 0.8503 | 0.1 | 0 |
| Naive Bayes (TF-IDF) | 0.7956 | 0.8584 | 0.1 | 0 |
| gbert-base (fine-tuned) | *pending* | *pending* | ~50 | 0 |
| GPT-4o-mini (zero-shot) | 0.7517 | 0.6700 | 966 | 0.19 |
| GPT-4o (zero-shot) | 0.6808 | 0.5850 | 1050 | 3.22 |
| GPT-4o (few-shot) | 0.5359 | 0.4450 | 1878 | 2.80 |
| GPT-4o-mini (few-shot) | 0.5054 | 0.4200 | 1889 | 0.17 |

**Key findings:**
- Classical ML with TF-IDF features outperforms zero-shot LLMs on this dataset
- SVM achieves the best F1 (0.856) with sub-millisecond latency and zero marginal cost
- Few-shot prompting surprisingly *hurts* LLM performance vs. zero-shot
- GPT-4o is not significantly better than GPT-4o-mini for this task but costs 16x more
- Fine-tuned BERT is expected to be the overall best (pending Colab training)

## Project Structure

```
customer-feedback-intelligence/
├── src/
│   ├── data/              # Dataset loading + preprocessing
│   ├── models/            # Classical ML, BERT, LLM classifiers
│   ├── training/          # Training orchestration + Optuna tuning
│   ├── evaluation/        # Metrics, comparison, error analysis
│   ├── inference/         # Production predictor
│   └── api.py             # FastAPI REST API
├── config/                # Central configuration
├── notebooks/             # Jupyter/Colab notebooks
├── tests/                 # pytest test suite (39 tests)
├── results/metrics/       # Saved evaluation metrics (JSON)
├── app.py                 # Streamlit dashboard
├── run_training.py        # CLI: train models
└── run_evaluation.py      # CLI: evaluate + compare models
```

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train classical models on German Sentiment
python run_training.py --model all --dataset german

# Evaluate LLM (requires Azure OpenAI key in .env)
python run_evaluation.py --model llm --deployment gpt-4o-mini --mode zero_shot

# Compare all models
python run_evaluation.py --model compare

# Run tests
pytest tests/ -v

# Launch dashboard
streamlit run app.py

# Start API
uvicorn src.api:app --reload
```

## Architecture

### Models

1. **Classical ML** (`src/models/classical.py`): TF-IDF vectorization (unigrams + bigrams, 50K features) with Logistic Regression, LinearSVC, or Multinomial Naive Bayes. Hyperparameter tuning via Optuna.

2. **Fine-tuned BERT** (`src/models/bert_classifier.py`): `deepset/gbert-base` (110M params) fine-tuned with HuggingFace Trainer. Training notebook for Google Colab included.

3. **LLM Zero/Few-Shot** (`src/models/llm_classifier.py`): Azure OpenAI GPT-4o and GPT-4o-mini with structured JSON output. Includes retry logic, cost tracking, and response caching.

### Evaluation

All models share the same evaluation pipeline (`src/evaluation/metrics.py`):
- Accuracy, Weighted F1, Macro F1, Precision, Recall
- Confusion Matrix (3x3)
- ROC-AUC (One-vs-Rest)
- Inference latency benchmarks
- API cost tracking (LLM models)

### Datasets

- **German Sentiment** (primary): 6.4K train / 772 val / 1.5K test, 3-class
- **Yelp Reviews** (secondary): 650K English reviews, 5-star mapped to 3-class

## API

```bash
# Predict single text
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Das Produkt ist ausgezeichnet!", "model": "classical"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Sehr gut!", "Schlecht."], "model": "classical"}'
```

## Tech Stack

- **ML**: scikit-learn, transformers, torch, optuna
- **Data**: pandas, datasets (HuggingFace)
- **API**: FastAPI, pydantic
- **Dashboard**: Streamlit, Plotly
- **LLM**: openai (Azure OpenAI SDK)
- **Tracking**: MLflow
- **Testing**: pytest (39 tests)
- **CI/CD**: GitHub Actions

## License

MIT
