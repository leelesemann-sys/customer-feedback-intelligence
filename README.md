# Customer Feedback Intelligence

A systematic comparison of three ML approaches for sentiment classification: **Classical ML** (TF-IDF + traditional classifiers), **Fine-tuned BERT**, and **LLM-based** (Azure OpenAI GPT-4o zero/few-shot).

Evaluated on two datasets: [German Sentiment](https://huggingface.co/datasets/sepidmnorozy/German_sentiment) (8.7K samples, 3-class) and [Yelp Reviews](https://huggingface.co/datasets/Yelp/yelp_review_full) (650K English reviews, 5-star → 3-class).

## Results

### German Sentiment (1,490 test samples)

All models evaluated on the **same test set** (stratified split, `seed=42`).

| Model | F1 (weighted) | Accuracy | Latency (ms/sample) | Cost/1K ($) |
|-------|:---:|:---:|:---:|:---:|
| **gbert-base (fine-tuned)** | **0.9119** | **0.9128** | 1.3 | 0 |
| SVM (TF-IDF) | 0.8562 | 0.8725 | 0.2 | 0 |
| Logistic Regression (TF-IDF) | 0.8505 | 0.8503 | 0.1 | 0 |
| Naive Bayes (TF-IDF) | 0.7956 | 0.8584 | 0.1 | 0 |
| GPT-4o-mini (zero-shot) | 0.7517 | 0.6700 | 966 | 0.19 |
| GPT-4o (zero-shot) | 0.6808 | 0.5850 | 1050 | 3.22 |
| GPT-4o (few-shot) | 0.5359 | 0.4450 | 1878 | 2.80 |
| GPT-4o-mini (few-shot) | 0.5054 | 0.4200 | 1889 | 0.17 |

### Yelp Reviews — English (5,000 test samples)

5-star ratings mapped to 3-class (1-2★ negative, 3★ neutral, 4-5★ positive). 10K stratified train subset.

| Model | F1 (weighted) | Accuracy | ROC-AUC | Latency (ms/sample) |
|-------|:---:|:---:|:---:|:---:|
| **bert-base-uncased (fine-tuned)** | **0.7607** | **0.7576** | **0.9135** | 1.4 |
| Logistic Regression (TF-IDF) | 0.7431 | 0.7393 | 0.9022 | 0.4 |
| SVM (TF-IDF) | 0.7375 | 0.7574 | 0.9028 | 0.3 |
| Naive Bayes (TF-IDF) | 0.6340 | 0.7129 | 0.8671 | 0.3 |

### Key findings

- **Fine-tuned BERT wins on both datasets**: German F1=0.912 (+5.6pp vs. best classical), Yelp F1=0.761 (+1.8pp)
- Classical ML with TF-IDF features outperforms zero-shot LLMs on the German dataset
- On Yelp, BERT's advantage over classical ML shrinks — Logistic Regression (F1=0.743) nearly matches BERT (F1=0.761)
- Few-shot prompting surprisingly *hurts* LLM performance vs. zero-shot
- GPT-4o is not significantly better than GPT-4o-mini for this task but costs 16x more
- BERT achieves near-real-time latency (1.3ms/sample on T4 GPU) at zero marginal cost
- Neutral class is hardest across all models on Yelp (BERT neutral F1=0.496, LogReg neutral F1=0.489) due to class imbalance (~20%)

## Methodology

### Evaluation rigor

- **Identical test sets**: All models evaluated on the exact same test split (stratified sampling with `seed=42`)
- **Multi-seed evaluation**: Classical ML supports training across 3 seeds (42, 123, 456) to report mean +/- std
- **Learning curves**: Validate that 10K training samples are sufficient (F1 plateaus beyond 5K for most classifiers)
- **Consistent label mapping**: Yelp 5-star ratings mapped to 3-class (1-2: negative, 3: neutral, 4-5: positive) following Zhang et al. (2015). This is a deliberate design choice for consistency with the German Sentiment dataset (also 3-class). Trade-off: star-3 is only ~20% of Yelp data, creating class imbalance.

### Statistical robustness

```bash
# Train with 3 seeds and report mean +/- std
python run_training.py --model classical --classifier svm --dataset german --multi-seed

# Compute learning curve (F1 at 500/1K/2K/5K/10K training samples)
python run_training.py --model classical --classifier svm --dataset german --learning-curve
```

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

2. **Fine-tuned BERT** (`src/models/bert_classifier.py`): `deepset/gbert-base` (German, 110M params) and `bert-base-uncased` (English, 110M params) fine-tuned with HuggingFace Trainer. Training notebooks for Google Colab included.

3. **LLM Zero/Few-Shot** (`src/models/llm_classifier.py`): Azure OpenAI GPT-4o and GPT-4o-mini with structured JSON output. Includes retry logic, cost tracking, and response caching.

### Evaluation

All models share the same evaluation pipeline (`src/evaluation/metrics.py`):
- Accuracy, Weighted F1, Macro F1, Precision, Recall
- Confusion Matrix (3x3)
- ROC-AUC (One-vs-Rest)
- Inference latency benchmarks
- API cost tracking (LLM models)
- Multi-seed aggregation (mean +/- std)
- Learning curves

### Datasets

- **German Sentiment** (primary): 6.4K train / 772 val / 1.5K test, 3-class
- **Yelp Reviews** (secondary): 650K English reviews, 5-star mapped to 3-class (10K train / 5K test subsets)

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
