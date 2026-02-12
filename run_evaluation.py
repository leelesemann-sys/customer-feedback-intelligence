"""
CLI for evaluating sentiment models (including LLM zero/few-shot).

Usage:
    python run_evaluation.py --model llm --dataset german --deployment gpt-4o-mini --mode zero_shot
    python run_evaluation.py --model llm --dataset german --deployment gpt-4o-mini --mode few_shot
    python run_evaluation.py --model llm --dataset german --deployment gpt-4o --mode zero_shot
    python run_evaluation.py --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_dataset(dataset_name: str, max_samples: int | None = None):
    """Load test split of the specified dataset."""
    if dataset_name == "german":
        from src.data.loader import load_german_sentiment
        return load_german_sentiment(max_train_samples=None)
    elif dataset_name == "yelp":
        from src.data.loader import load_yelp_reviews
        return load_yelp_reviews(max_train_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_llm(
    deployment: str = "gpt-4o-mini",
    mode: str = "zero_shot",
    dataset_name: str = "german",
    max_test_samples: int | None = None,
) -> None:
    """Evaluate an LLM model on the test set."""
    from src.data.preprocessor import TextPreprocessor
    from src.evaluation.metrics import compute_metrics
    from src.models.llm_classifier import LLMSentimentModel
    from config.config import RESULTS_DIR

    # Load data
    splits = _load_dataset(dataset_name)
    preprocessor = TextPreprocessor()
    test_df = preprocessor.preprocess_dataframe(splits["test"])

    test_texts = test_df["text_clean"].tolist()
    test_labels = test_df["label"].tolist()

    if max_test_samples:
        test_texts = test_texts[:max_test_samples]
        test_labels = test_labels[:max_test_samples]

    logger.info("Evaluating %s (%s) on %d test samples...", deployment, mode, len(test_texts))

    # Create model
    model = LLMSentimentModel(deployment=deployment, mode=mode)

    # Predict with timing
    start = time.perf_counter()
    predictions = model.predict(test_texts)
    duration = time.perf_counter() - start

    probas = model.predict_proba(test_texts)

    # Compute metrics
    metrics = compute_metrics(np.array(test_labels), predictions, probas)
    cost = model.get_cost()

    # Latency
    latency = {
        "total_ms": round(duration * 1000, 2),
        "per_sample_ms": round((duration / len(test_texts)) * 1000, 2),
        "samples_per_second": round(len(test_texts) / duration, 1),
    }

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS: %s (%s) on %s", deployment, mode, dataset_name)
    logger.info("=" * 60)
    logger.info("  Accuracy:     %.4f", metrics["accuracy"])
    logger.info("  F1 weighted:  %.4f", metrics["f1_weighted"])
    logger.info("  F1 macro:     %.4f", metrics["f1_macro"])
    logger.info("  Precision:    %.4f", metrics["precision_weighted"])
    logger.info("  Recall:       %.4f", metrics["recall_weighted"])
    if metrics.get("roc_auc_weighted"):
        logger.info("  ROC-AUC:      %.4f", metrics["roc_auc_weighted"])
    logger.info("  Latency:      %.1f ms/sample", latency["per_sample_ms"])
    logger.info("  Total cost:   $%.4f", cost["total_cost_usd"])
    logger.info("  Tokens:       %d in, %d out", cost["input_tokens"], cost["output_tokens"])

    # Save results
    all_results = {
        "model": model.name,
        "deployment": deployment,
        "mode": mode,
        "dataset": dataset_name,
        "n_test_samples": len(test_texts),
        "duration_s": round(duration, 2),
        "test": {k: v for k, v in metrics.items() if k != "classification_report"},
        "latency": latency,
        "cost": cost,
        "test_classification_report": metrics.get("classification_report"),
    }

    metrics_path = RESULTS_DIR / "metrics" / f"{model.name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Metrics saved to %s", metrics_path)


def compare_models() -> None:
    """Load all saved metrics and print a comparison table."""
    from config.config import RESULTS_DIR

    metrics_dir = RESULTS_DIR / "metrics"
    if not metrics_dir.exists():
        logger.error("No metrics found in %s", metrics_dir)
        return

    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        test = data.get("test", {})
        latency = data.get("latency", {})
        cost = data.get("cost", {})

        rows.append({
            "Model": data.get("model", f.stem),
            "F1 (weighted)": test.get("f1_weighted", 0),
            "Accuracy": test.get("accuracy", 0),
            "Latency (ms/sample)": latency.get("per_sample_ms", 0),
            "Cost/1K ($)": cost.get("cost_per_1k_predictions", 0) if cost else 0,
        })

    if not rows:
        logger.info("No metrics files found.")
        return

    # Print table
    header = f"{'Model':<40} {'F1':>8} {'Acc':>8} {'Lat(ms)':>10} {'$/1K':>8}"
    logger.info("=" * 78)
    logger.info(header)
    logger.info("-" * 78)
    for r in sorted(rows, key=lambda x: x["F1 (weighted)"], reverse=True):
        logger.info(
            f"{r['Model']:<40} {r['F1 (weighted)']:>8.4f} {r['Accuracy']:>8.4f} "
            f"{r['Latency (ms/sample)']:>10.1f} {r['Cost/1K ($)']:>8.4f}"
        )
    logger.info("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment models")
    parser.add_argument(
        "--model",
        choices=["llm", "compare"],
        required=True,
    )
    parser.add_argument("--dataset", choices=["yelp", "german"], default="german")
    parser.add_argument("--deployment", default="gpt-4o-mini")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--max-test-samples", type=int, default=None)

    args = parser.parse_args()

    if args.model == "llm":
        evaluate_llm(
            deployment=args.deployment,
            mode=args.mode,
            dataset_name=args.dataset,
            max_test_samples=args.max_test_samples,
        )
    elif args.model == "compare":
        compare_models()


if __name__ == "__main__":
    main()
