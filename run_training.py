"""
CLI for training sentiment models.

Usage:
    python run_training.py --model classical --dataset yelp
    python run_training.py --model classical --dataset german
    python run_training.py --model classical --classifier logistic_regression --optimize
    python run_training.py --model all --dataset yelp --max-train-samples 10000
    python run_training.py --model classical --multi-seed --dataset german
    python run_training.py --model classical --learning-curve --dataset german
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_dataset(dataset_name: str, max_train_samples: int | None = None):
    """Load the specified dataset.

    For Yelp, test set defaults to 5K stratified samples for consistency
    across all model evaluations (Classical ML, BERT notebook, LLM).
    """
    if dataset_name == "yelp":
        from src.data.loader import load_yelp_reviews
        return load_yelp_reviews(
            max_train_samples=max_train_samples,
            max_test_samples=5000,
        )
    elif dataset_name == "german":
        from src.data.loader import load_german_sentiment
        return load_german_sentiment(max_train_samples=max_train_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_classical(
    classifier_name: str = "logistic_regression",
    dataset_name: str = "yelp",
    optimize: bool = False,
    max_train_samples: int | None = None,
) -> None:
    """Train a classical ML model."""
    from src.data.preprocessor import TextPreprocessor
    from src.models.classical import ClassicalSentimentModel
    from src.training.trainer import setup_mlflow, train_and_evaluate

    setup_mlflow()

    # Load data
    splits = _load_dataset(dataset_name, max_train_samples)
    preprocessor = TextPreprocessor()

    for split_name in splits:
        splits[split_name] = preprocessor.preprocess_dataframe(splits[split_name])

    train_texts = splits["train"]["text_clean"].tolist()
    train_labels = splits["train"]["label"].tolist()
    val_texts = splits["val"]["text_clean"].tolist()
    val_labels = splits["val"]["label"].tolist()
    test_texts = splits["test"]["text_clean"].tolist()
    test_labels = splits["test"]["label"].tolist()

    if optimize:
        from src.training.hyperparameter import optimize_classical

        logger.info("Running hyperparameter optimization for %s...", classifier_name)
        result = optimize_classical(
            classifier_name, train_texts, train_labels, val_texts, val_labels
        )
        model = result["best_model"]
        extra_params = result["best_params"]
    else:
        model = ClassicalSentimentModel(classifier_name=classifier_name)
        extra_params = {"classifier": classifier_name, "dataset": dataset_name}

    train_and_evaluate(
        model=model,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        extra_params=extra_params,
    )


def train_all_classical(
    dataset_name: str = "yelp",
    max_train_samples: int | None = None,
) -> None:
    """Train all three classical models."""
    for clf in ["logistic_regression", "svm", "naive_bayes"]:
        logger.info("=" * 60)
        logger.info("Training %s", clf)
        logger.info("=" * 60)
        train_classical(
            classifier_name=clf,
            dataset_name=dataset_name,
            max_train_samples=max_train_samples,
        )


def train_multi_seed(
    classifier_name: str = "logistic_regression",
    dataset_name: str = "german",
    max_train_samples: int | None = None,
) -> None:
    """Train a classical model across multiple seeds for statistical robustness."""
    import json
    from functools import partial
    from config.config import EVAL_SEEDS, RESULTS_DIR
    from src.models.classical import ClassicalSentimentModel
    from src.training.trainer import train_multi_seed as _train_multi_seed

    if dataset_name == "german":
        from src.data.loader import load_german_sentiment
        loader = partial(load_german_sentiment, max_train_samples=max_train_samples)
    else:
        from src.data.loader import load_yelp_reviews
        loader = partial(load_yelp_reviews, max_train_samples=max_train_samples)

    result = _train_multi_seed(
        model_factory=lambda: ClassicalSentimentModel(classifier_name=classifier_name),
        dataset_loader=loader,
        seeds=EVAL_SEEDS,
        dataset_name=dataset_name,
        max_train_samples=max_train_samples,
    )

    # Save aggregated results
    out_path = RESULTS_DIR / "metrics" / f"multi_seed_{classifier_name}_{dataset_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Multi-seed results saved to %s", out_path)


def run_learning_curve(
    classifier_name: str = "logistic_regression",
    dataset_name: str = "german",
) -> None:
    """Compute learning curve for a classical model."""
    import json
    from config.config import RESULTS_DIR
    from src.data.preprocessor import TextPreprocessor
    from src.models.classical import ClassicalSentimentModel
    from src.training.trainer import compute_learning_curve

    if dataset_name == "german":
        from src.data.loader import load_german_sentiment
        splits = load_german_sentiment()
    else:
        from src.data.loader import load_yelp_reviews
        splits = load_yelp_reviews(max_train_samples=50000)

    preprocessor = TextPreprocessor()
    for split_name in splits:
        splits[split_name] = preprocessor.preprocess_dataframe(splits[split_name])

    train_sizes = [500, 1000, 2000, 5000, 10000]
    if dataset_name == "yelp":
        train_sizes.append(25000)
        train_sizes.append(50000)

    logger.info("Computing learning curve for %s on %s...", classifier_name, dataset_name)
    results = compute_learning_curve(
        model_factory=lambda: ClassicalSentimentModel(classifier_name=classifier_name),
        train_texts=splits["train"]["text_clean"].tolist(),
        train_labels=splits["train"]["label"].tolist(),
        test_texts=splits["test"]["text_clean"].tolist(),
        test_labels=splits["test"]["label"].tolist(),
        train_sizes=train_sizes,
    )

    out_path = RESULTS_DIR / "metrics" / f"learning_curve_{classifier_name}_{dataset_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Learning curve saved to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Train sentiment models")
    parser.add_argument(
        "--model",
        choices=["classical", "bert", "llm", "all"],
        required=True,
        help="Model type to train (llm = evaluate via run_evaluation.py)",
    )
    parser.add_argument(
        "--dataset",
        choices=["yelp", "german"],
        default="yelp",
        help="Dataset to train on (default: yelp)",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "svm", "naive_bayes"],
        default="logistic_regression",
        help="Classical classifier to use",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training set size (for faster iteration)",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Train across 3 seeds and report mean +/- std",
    )
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="Compute learning curve (F1 at different training set sizes)",
    )

    args = parser.parse_args()

    if args.learning_curve and args.model == "classical":
        run_learning_curve(
            classifier_name=args.classifier,
            dataset_name=args.dataset,
        )
    elif args.multi_seed and args.model == "classical":
        train_multi_seed(
            classifier_name=args.classifier,
            dataset_name=args.dataset,
            max_train_samples=args.max_train_samples,
        )
    elif args.model == "classical":
        train_classical(
            classifier_name=args.classifier,
            dataset_name=args.dataset,
            optimize=args.optimize,
            max_train_samples=args.max_train_samples,
        )
    elif args.model == "all":
        train_all_classical(
            dataset_name=args.dataset,
            max_train_samples=args.max_train_samples,
        )
    elif args.model == "bert":
        logger.info("BERT training: use notebooks/04_bert_finetuning.ipynb on Google Colab.")
        sys.exit(1)
    elif args.model == "llm":
        logger.info("LLM models don't require training. Use run_evaluation.py instead:")
        logger.info("  python run_evaluation.py --model llm --deployment gpt-4o-mini --mode zero_shot")
        sys.exit(0)


if __name__ == "__main__":
    main()
