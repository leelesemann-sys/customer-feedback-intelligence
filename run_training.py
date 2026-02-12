"""
CLI for training sentiment models.

Usage:
    python run_training.py --model classical --dataset yelp
    python run_training.py --model classical --dataset german
    python run_training.py --model classical --classifier logistic_regression --optimize
    python run_training.py --model all --dataset yelp --max-train-samples 10000
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
    """Load the specified dataset."""
    if dataset_name == "yelp":
        from src.data.loader import load_yelp_reviews
        return load_yelp_reviews(max_train_samples=max_train_samples)
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


def main():
    parser = argparse.ArgumentParser(description="Train sentiment models")
    parser.add_argument(
        "--model",
        choices=["classical", "bert", "all"],
        required=True,
        help="Model type to train",
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

    args = parser.parse_args()

    if args.model == "classical":
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
        logger.info("BERT training not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
