# Customer Feedback Intelligence — Sentiment Analysis Benchmark

> **Sprache:** [English](README.md) | Deutsch

[![CI](https://github.com/leelesemann-sys/customer-feedback-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/leelesemann-sys/customer-feedback-intelligence/actions/workflows/ci.yml)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-feedback-intelligence.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-End **Sentiment-Analyse**-Benchmark, der drei ML-Ansätze auf Kundenfeedback-Daten vergleicht: **Klassisches ML** (TF-IDF + SVM/LogReg/NB), **Fine-tuned BERT** (HuggingFace Transformers) und **LLM Zero/Few-Shot** (Azure OpenAI GPT-4o). Beinhaltet Trainingspipelines, Evaluationsframework, REST API und interaktives Dashboard.

Evaluiert auf zwei Datensätzen: [German Sentiment](https://huggingface.co/datasets/sepidmnorozy/German_sentiment) (8,7K Samples, 3 Klassen) und [Yelp Reviews](https://huggingface.co/datasets/Yelp/yelp_review_full) (650K englische Reviews, 5-Sterne → 3 Klassen).

> **[Live-Dashboard](https://customer-feedback-intelligence.streamlit.app)** | **[GitHub Repo](https://github.com/leelesemann-sys/customer-feedback-intelligence)** — Interaktiver Ergebnis-Explorer mit Modellvergleich, Konfusionsmatrizen, Fehleranalyse und Live-Sentiment-Klassifikator.

## Ergebnisse

### German Sentiment (1.490 Testsamples)

Alle Modelle auf **demselben Testset** evaluiert (stratifizierter Split, `seed=42`).

| Modell | F1 (gewichtet) | Accuracy | Latenz (ms/Sample) | Kosten/1K ($) |
|--------|:---:|:---:|:---:|:---:|
| **gbert-base (fine-tuned)** | **0.9119** | **0.9128** | 1.3 | 0 |
| SVM (TF-IDF) | 0.8562 | 0.8725 | 0.2 | 0 |
| Logistic Regression (TF-IDF) | 0.8505 | 0.8503 | 0.1 | 0 |
| Naive Bayes (TF-IDF) | 0.7956 | 0.8584 | 0.1 | 0 |
| GPT-4o-mini (zero-shot) | 0.7517 | 0.6700 | 966 | 0.19 |
| GPT-4o (zero-shot) | 0.6808 | 0.5850 | 1050 | 3.22 |
| GPT-4o (few-shot) | 0.5359 | 0.4450 | 1878 | 2.80 |
| GPT-4o-mini (few-shot) | 0.5054 | 0.4200 | 1889 | 0.17 |

### Yelp Reviews — Englisch (5.000 Testsamples)

5-Sterne-Bewertungen auf 3 Klassen abgebildet (1–2 Sterne negativ, 3 Sterne neutral, 4–5 Sterne positiv). 10K stratifiziertes Trainingssubset.

| Modell | F1 (gewichtet) | Accuracy | ROC-AUC | Latenz (ms/Sample) |
|--------|:---:|:---:|:---:|:---:|
| **bert-base-uncased (fine-tuned)** | **0.7607** | **0.7576** | **0.9135** | 1.4 |
| Logistic Regression (TF-IDF) | 0.7431 | 0.7393 | 0.9022 | 0.4 |
| SVM (TF-IDF) | 0.7375 | 0.7574 | 0.9028 | 0.3 |
| Naive Bayes (TF-IDF) | 0.6340 | 0.7129 | 0.8671 | 0.3 |

### Zentrale Erkenntnisse

- **Fine-tuned BERT gewinnt auf beiden Datensätzen**: German F1=0,912 (+5,6 Pp vs. bestes klassisches Modell), Yelp F1=0,761 (+1,8 Pp)
- Klassisches ML mit TF-IDF-Features übertrifft Zero-Shot-LLMs auf dem deutschen Datensatz
- Bei Yelp schrumpft der Vorsprung von BERT gegenüber klassischem ML — Logistic Regression (F1=0,743) kommt BERT (F1=0,761) nahe
- Few-Shot-Prompting verschlechtert die LLM-Performance überraschenderweise gegenüber Zero-Shot
- GPT-4o ist für diese Aufgabe nicht signifikant besser als GPT-4o-mini, kostet aber 16x mehr
- BERT erreicht Echtzeit-Latenz (1,3 ms/Sample auf T4 GPU) bei null Grenzkosten
- Die neutrale Klasse ist am schwierigsten über alle Modelle auf Yelp (BERT neutral F1=0,496, LogReg neutral F1=0,489) aufgrund von Klassenungleichgewicht (~20 %)

## Methodik

### Evaluationsstrenge

- **Identische Testsets**: Alle Modelle auf exakt demselben Test-Split evaluiert (stratifiziertes Sampling mit `seed=42`)
- **Multi-Seed-Evaluation**: Klassisches ML unterstützt Training über 3 Seeds (42, 123, 456) zur Angabe von Mittelwert +/- Std
- **Lernkurven**: Validierung, dass 10K Trainingssamples ausreichen (F1 plateaut ab 5K für die meisten Klassifikatoren)
- **Konsistentes Label-Mapping**: Yelp 5-Sterne-Bewertungen auf 3 Klassen abgebildet (1–2: negativ, 3: neutral, 4–5: positiv) nach Zhang et al. (2015). Dies ist eine bewusste Designentscheidung für Konsistenz mit dem German Sentiment Datensatz (ebenfalls 3 Klassen). Trade-off: 3-Sterne umfasst nur ~20 % der Yelp-Daten, was Klassenungleichgewicht erzeugt.

### Statistische Robustheit

```bash
# Training mit 3 Seeds und Mittelwert +/- Std ausgeben
python run_training.py --model classical --classifier svm --dataset german --multi-seed

# Lernkurve berechnen (F1 bei 500/1K/2K/5K/10K Trainingssamples)
python run_training.py --model classical --classifier svm --dataset german --learning-curve
```

## Projektstruktur

```
customer-feedback-intelligence/
├── src/
│   ├── data/              # Dataset-Loading + Vorverarbeitung
│   ├── models/            # Klassisches ML, BERT, LLM-Klassifikatoren
│   ├── training/          # Trainings-Orchestrierung + Optuna-Tuning
│   ├── evaluation/        # Metriken, Vergleich, Fehleranalyse
│   ├── inference/         # Produktions-Predictor
│   └── api.py             # FastAPI REST API
├── config/                # Zentrale Konfiguration
├── notebooks/             # Jupyter/Colab-Notebooks
├── tests/                 # pytest-Testsuite (39 Tests)
├── results/metrics/       # Gespeicherte Evaluationsmetriken (JSON)
├── app.py                 # Streamlit Dashboard
├── run_training.py        # CLI: Modelle trainieren
└── run_evaluation.py      # CLI: Evaluieren + Modelle vergleichen
```

## Schnellstart

```bash
# Setup
python -m venv venv
source venv/bin/activate  # oder venv\Scripts\activate unter Windows
pip install -r requirements-full.txt

# Klassische Modelle auf German Sentiment trainieren
python run_training.py --model all --dataset german

# LLM evaluieren (erfordert Azure OpenAI Key in .env)
python run_evaluation.py --model llm --deployment gpt-4o-mini --mode zero_shot

# Alle Modelle vergleichen
python run_evaluation.py --model compare

# Tests ausführen
pytest tests/ -v

# Dashboard starten
streamlit run app.py

# API starten
uvicorn src.api:app --reload
```

## Architektur

### Modelle

1. **Klassisches ML** (`src/models/classical.py`): TF-IDF-Vektorisierung (Unigramme + Bigramme, 50K Features) mit Logistic Regression, LinearSVC oder Multinomial Naive Bayes. Hyperparameter-Tuning via Optuna.

2. **Fine-tuned BERT** (`src/models/bert_classifier.py`): `deepset/gbert-base` (Deutsch, 110M Parameter) und `bert-base-uncased` (Englisch, 110M Parameter), fine-tuned mit HuggingFace Trainer. Trainings-Notebooks für Google Colab enthalten.

3. **LLM Zero/Few-Shot** (`src/models/llm_classifier.py`): Azure OpenAI GPT-4o und GPT-4o-mini mit strukturiertem JSON-Output. Enthält Retry-Logik, Kostentracking und Response-Caching.

### Evaluation

Alle Modelle teilen dieselbe Evaluationspipeline (`src/evaluation/metrics.py`):
- Accuracy, Weighted F1, Macro F1, Precision, Recall
- Konfusionsmatrix (3x3)
- ROC-AUC (One-vs-Rest)
- Inferenz-Latenz-Benchmarks
- API-Kostentracking (LLM-Modelle)
- Multi-Seed-Aggregation (Mittelwert +/- Std)
- Lernkurven

### Datensätze

- **German Sentiment** (primär): 6,4K Train / 772 Val / 1,5K Test, 3 Klassen
- **Yelp Reviews** (sekundär): 650K englische Reviews, 5-Sterne auf 3 Klassen abgebildet (10K Train / 5K Test Subsets)

## API

```bash
# Einzelnen Text vorhersagen
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Das Produkt ist ausgezeichnet!", "model": "classical"}'

# Batch-Vorhersage
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Sehr gut!", "Schlecht."], "model": "classical"}'
```

## Tech Stack

- **ML**: scikit-learn, transformers, torch, optuna
- **Daten**: pandas, datasets (HuggingFace)
- **API**: FastAPI, pydantic
- **Dashboard**: Streamlit, Plotly
- **LLM**: openai (Azure OpenAI SDK)
- **Tracking**: MLflow
- **Testing**: pytest (39 Tests)
- **CI/CD**: GitHub Actions

## Lizenz

MIT
