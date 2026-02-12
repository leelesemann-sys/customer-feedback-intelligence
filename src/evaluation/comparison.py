"""
Multi-model comparison framework.

Loads metrics from all models and produces comparison tables and visualizations.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from config.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def load_all_metrics() -> list[dict]:
    """Load all saved model metrics from results/metrics/."""
    metrics_dir = RESULTS_DIR / "metrics"
    if not metrics_dir.exists():
        return []

    all_metrics = []
    for f in sorted(metrics_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        all_metrics.append(data)

    return all_metrics


def build_comparison_table(metrics: list[dict] | None = None) -> pd.DataFrame:
    """Build a comparison DataFrame from all model metrics.

    Returns:
        DataFrame with one row per model and columns for key metrics.
    """
    if metrics is None:
        metrics = load_all_metrics()

    rows = []
    for m in metrics:
        test = m.get("test", {})
        latency = m.get("latency", {})
        cost = m.get("cost", {})
        train = m.get("train", {})

        rows.append({
            "model": m.get("model", "unknown"),
            "accuracy": test.get("accuracy", 0),
            "f1_weighted": test.get("f1_weighted", 0),
            "f1_macro": test.get("f1_macro", 0),
            "precision_weighted": test.get("precision_weighted", 0),
            "recall_weighted": test.get("recall_weighted", 0),
            "roc_auc_weighted": test.get("roc_auc_weighted"),
            "latency_ms_per_sample": latency.get("per_sample_ms", 0),
            "cost_per_1k_usd": cost.get("cost_per_1k_predictions", 0) if cost else 0,
            "train_duration_s": train.get("train_duration_s", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("f1_weighted", ascending=False).reset_index(drop=True)

    return df


def print_comparison_table(df: pd.DataFrame | None = None) -> str:
    """Format comparison table as a readable string."""
    if df is None:
        df = build_comparison_table()

    if df.empty:
        return "No model metrics found."

    display_cols = ["model", "f1_weighted", "accuracy", "f1_macro", "latency_ms_per_sample", "cost_per_1k_usd"]
    display_df = df[display_cols].copy()
    display_df.columns = ["Model", "F1 (weighted)", "Accuracy", "F1 (macro)", "Latency (ms)", "Cost/1K ($)"]

    return display_df.to_string(index=False, float_format="%.4f")


def get_best_model(metric: str = "f1_weighted") -> dict | None:
    """Return the metrics dict for the best model by a given metric."""
    metrics = load_all_metrics()
    if not metrics:
        return None

    return max(metrics, key=lambda m: m.get("test", {}).get(metric, 0))
