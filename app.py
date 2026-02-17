"""
Streamlit Dashboard for Customer Feedback Intelligence.

Compares Classical ML, Fine-tuned BERT, and LLM approaches
across German Sentiment and Yelp Reviews datasets.

Usage:
    streamlit run app.py
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from config.config import LABEL_NAMES, RESULTS_DIR


st.set_page_config(
    page_title="Customer Feedback Intelligence — Sentiment Analysis: BERT vs Classical ML vs LLM",
    page_icon="📊",
    layout="wide",
    menu_items={
        "About": (
            "## Customer Feedback Intelligence\n"
            "Sentiment analysis benchmark comparing Classical ML, Fine-tuned BERT, "
            "and GPT-4o on German and English customer feedback.\n\n"
            "**[GitHub Repository](https://github.com/leelesemann-sys/customer-feedback-intelligence)**"
        ),
        "Report a bug": "https://github.com/leelesemann-sys/customer-feedback-intelligence/issues",
    },
)

# ---------- Data Loading ----------

DATASET_FILES = {
    "German Sentiment": {
        "classical_svm": "classical_svm.json",
        "classical_logistic_regression": "classical_logistic_regression.json",
        "classical_naive_bayes": "classical_naive_bayes.json",
        "bert_finetuned": "bert_finetuned_metrics.json",
        "llm_gpt-4o-mini_zero_shot": "llm_gpt-4o-mini_zero_shot.json",
        "llm_gpt-4o_zero_shot": "llm_gpt-4o_zero_shot.json",
        "llm_gpt-4o_few_shot": "llm_gpt-4o_few_shot.json",
        "llm_gpt-4o-mini_few_shot": "llm_gpt-4o-mini_few_shot.json",
    },
    "Yelp Reviews": {
        "classical_svm": "classical_svm_yelp.json",
        "classical_logistic_regression": "classical_logistic_regression_yelp.json",
        "classical_naive_bayes": "classical_naive_bayes_yelp.json",
        "bert_finetuned": "bert_yelp_finetuned_metrics.json",
    },
}

DISPLAY_NAMES = {
    "classical_svm": "SVM (TF-IDF)",
    "classical_logistic_regression": "Logistic Regression (TF-IDF)",
    "classical_naive_bayes": "Naive Bayes (TF-IDF)",
    "bert_finetuned": "BERT (fine-tuned)",
    "llm_gpt-4o-mini_zero_shot": "GPT-4o-mini (zero-shot)",
    "llm_gpt-4o_zero_shot": "GPT-4o (zero-shot)",
    "llm_gpt-4o_few_shot": "GPT-4o (few-shot)",
    "llm_gpt-4o-mini_few_shot": "GPT-4o-mini (few-shot)",
}

MODEL_CATEGORIES = {
    "classical_svm": "Classical ML",
    "classical_logistic_regression": "Classical ML",
    "classical_naive_bayes": "Classical ML",
    "bert_finetuned": "Fine-tuned BERT",
    "llm_gpt-4o-mini_zero_shot": "LLM",
    "llm_gpt-4o_zero_shot": "LLM",
    "llm_gpt-4o_few_shot": "LLM",
    "llm_gpt-4o-mini_few_shot": "LLM",
}

CATEGORY_COLORS = {
    "Classical ML": "#3B82F6",
    "Fine-tuned BERT": "#10B981",
    "LLM": "#8B5CF6",
}


@st.cache_data
def load_metrics_for_dataset(dataset_name: str) -> list[dict]:
    """Load metrics for a specific dataset."""
    metrics_dir = RESULTS_DIR / "metrics"
    results = []
    for model_key, filename in DATASET_FILES.get(dataset_name, {}).items():
        fpath = metrics_dir / filename
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            data["_model_key"] = model_key
            data["_display_name"] = DISPLAY_NAMES.get(model_key, model_key)
            data["_category"] = MODEL_CATEGORIES.get(model_key, "Other")
            results.append(data)
    return results


@st.cache_data
def load_learning_curve() -> list[dict] | None:
    """Load learning curve data if available."""
    fpath = RESULTS_DIR / "metrics" / "learning_curve_svm_german.json"
    if fpath.exists():
        with open(fpath) as f:
            return json.load(f)
    return None


@st.cache_data
def load_multi_seed() -> dict | None:
    """Load multi-seed data if available."""
    fpath = RESULTS_DIR / "metrics" / "multi_seed_svm_german.json"
    if fpath.exists():
        with open(fpath) as f:
            return json.load(f)
    return None


def build_comparison_df(metrics: list[dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from metrics list."""
    rows = []
    for m in metrics:
        test = m.get("test", {})
        latency = m.get("latency", {})
        cost = m.get("cost", {})
        rows.append({
            "Model": m["_display_name"],
            "Category": m["_category"],
            "F1 (weighted)": test.get("f1_weighted", 0),
            "Accuracy": test.get("accuracy", 0),
            "F1 (macro)": test.get("f1_macro", 0),
            "Precision": test.get("precision_weighted", 0),
            "Recall": test.get("recall_weighted", 0),
            "ROC-AUC": test.get("roc_auc_weighted", 0) or 0,
            "Latency (ms)": latency.get("per_sample_ms", 0),
            "Cost/1K ($)": cost.get("cost_per_1k_predictions", 0) if cost else 0,
        })
    return pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)


# ---------- Sidebar ----------

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Comparison", "Error Analysis", "Methodology", "Live Demo", "Batch Analysis"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)]"
    "(https://github.com/leelesemann-sys/customer-feedback-intelligence)"
)
st.sidebar.caption("Built with Streamlit, scikit-learn, HuggingFace Transformers & Azure OpenAI")


# ==================== PAGE 1: Overview ====================
if page == "Overview":
    st.title("Customer Feedback Intelligence")
    st.markdown(
        "A systematic comparison of **Classical ML**, **Fine-tuned BERT**, and **LLM** approaches "
        "for sentiment classification across two datasets."
    )

    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "BERT", "F1 = 0.912")
    col2.metric("Datasets", "2", "German + Yelp")
    col3.metric("Models Compared", "8", "3 approaches")
    col4.metric("Test Samples", "6,490", "1.5K + 5K")

    st.markdown("---")

    # Side-by-side dataset results
    tab_german, tab_yelp = st.tabs(["🇩🇪 German Sentiment", "🇺🇸 Yelp Reviews"])

    with tab_german:
        metrics = load_metrics_for_dataset("German Sentiment")
        if metrics:
            df = build_comparison_df(metrics)

            fig = px.bar(
                df, x="Model", y="F1 (weighted)",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                title="German Sentiment — F1 Score by Model",
            )
            fig.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df[["Model", "Category", "F1 (weighted)", "Accuracy", "Latency (ms)", "Cost/1K ($)"]].style.format({
                    "F1 (weighted)": "{:.4f}", "Accuracy": "{:.4f}",
                    "Latency (ms)": "{:.1f}", "Cost/1K ($)": "{:.2f}",
                }).highlight_max(subset=["F1 (weighted)"], color="#d4edda"),
                use_container_width=True, hide_index=True,
            )

    with tab_yelp:
        metrics = load_metrics_for_dataset("Yelp Reviews")
        if metrics:
            df = build_comparison_df(metrics)

            fig = px.bar(
                df, x="Model", y="F1 (weighted)",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                title="Yelp Reviews — F1 Score by Model",
            )
            fig.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df[["Model", "Category", "F1 (weighted)", "Accuracy", "ROC-AUC", "Latency (ms)"]].style.format({
                    "F1 (weighted)": "{:.4f}", "Accuracy": "{:.4f}",
                    "ROC-AUC": "{:.4f}", "Latency (ms)": "{:.1f}",
                }).highlight_max(subset=["F1 (weighted)"], color="#d4edda"),
                use_container_width=True, hide_index=True,
            )


# ==================== PAGE 2: Model Comparison ====================
elif page == "Model Comparison":
    st.title("Model Comparison")

    dataset = st.selectbox("Dataset", list(DATASET_FILES.keys()))
    metrics = load_metrics_for_dataset(dataset)

    if not metrics:
        st.warning("No metrics found for this dataset.")
        st.stop()

    df = build_comparison_df(metrics)

    # Performance table
    st.subheader("Performance Overview")
    st.dataframe(
        df.style.format({
            "F1 (weighted)": "{:.4f}", "Accuracy": "{:.4f}",
            "F1 (macro)": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "ROC-AUC": "{:.4f}",
            "Latency (ms)": "{:.1f}", "Cost/1K ($)": "{:.4f}",
        }).highlight_max(
            subset=["F1 (weighted)", "Accuracy"], color="#d4edda"
        ).highlight_min(
            subset=["Latency (ms)"], color="#d4edda"
        ),
        use_container_width=True, hide_index=True,
    )

    # Charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("F1 Score Comparison")
        fig = px.bar(
            df, x="Model", y="F1 (weighted)",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Accuracy vs. Latency")
        fig = px.scatter(
            df, x="Latency (ms)", y="Accuracy",
            color="Category", size="F1 (weighted)",
            hover_name="Model",
            color_discrete_map=CATEGORY_COLORS,
            log_x=True,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.subheader("Multi-Metric Radar")
    radar_metrics = ["F1 (weighted)", "Accuracy", "Precision", "Recall", "F1 (macro)"]
    fig = go.Figure()
    for _, row in df.iterrows():
        values = [row[m] for m in radar_metrics]
        values.append(values[0])  # close polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            name=row["Model"],
            fill="toself",
            opacity=0.3,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1])),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(min(len(metrics), 3))
    for i, m in enumerate(metrics):
        cm = m.get("test", {}).get("confusion_matrix")
        if not cm:
            cm = m.get("confusion_matrix")
        if cm:
            with cols[i % 3]:
                st.markdown(f"**{m['_display_name']}**")
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=LABEL_NAMES,
                    y=LABEL_NAMES,
                    text_auto=True,
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE 3: Error Analysis ====================
elif page == "Error Analysis":
    st.title("Error Analysis")

    dataset = st.selectbox("Dataset", list(DATASET_FILES.keys()))
    metrics = load_metrics_for_dataset(dataset)

    if not metrics:
        st.warning("No metrics found.")
        st.stop()

    selected_name = st.selectbox(
        "Select model",
        [m["_display_name"] for m in metrics],
    )

    model_data = next((m for m in metrics if m["_display_name"] == selected_name), None)
    if not model_data:
        st.error("Model data not found.")
        st.stop()

    test = model_data.get("test", {})
    report = model_data.get("test_classification_report") or model_data.get("classification_report", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("F1 (weighted)", f"{test.get('f1_weighted', 0):.4f}")
    col2.metric("Accuracy", f"{test.get('accuracy', 0):.4f}")
    col3.metric("F1 (macro)", f"{test.get('f1_macro', 0):.4f}")

    # Per-class metrics
    if report:
        st.subheader("Per-Class Performance")
        class_rows = []
        for label in LABEL_NAMES:
            if label in report:
                class_rows.append({
                    "Class": label,
                    "Precision": report[label].get("precision", 0),
                    "Recall": report[label].get("recall", 0),
                    "F1": report[label].get("f1-score", 0),
                    "Support": int(report[label].get("support", 0)),
                })
        if class_rows:
            class_df = pd.DataFrame(class_rows)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    class_df.style.format({
                        "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col2:
                fig = px.bar(
                    class_df, x="Class", y=["Precision", "Recall", "F1"],
                    barmode="group", color_discrete_sequence=["#3B82F6", "#F59E0B", "#10B981"],
                )
                fig.update_layout(height=300, yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    cm = test.get("confusion_matrix") or model_data.get("confusion_matrix")
    if cm:
        st.subheader("Confusion Matrix")
        cm_array = np.array(cm)
        # Normalized confusion matrix
        cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Absolute Counts**")
            fig = px.imshow(
                cm, labels=dict(x="Predicted", y="True", color="Count"),
                x=LABEL_NAMES, y=LABEL_NAMES, text_auto=True,
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Normalized (row %)**")
            fig = px.imshow(
                cm_norm, labels=dict(x="Predicted", y="True", color="Rate"),
                x=LABEL_NAMES, y=LABEL_NAMES,
                text_auto=".1%", color_continuous_scale="Blues",
                zmin=0, zmax=1,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Cost info for LLM models
    cost = model_data.get("cost")
    if cost:
        st.subheader("API Cost")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost", f"${cost.get('total_cost_usd', 0):.4f}")
        c2.metric("Input Tokens", f"{cost.get('input_tokens', 0):,}")
        c3.metric("Output Tokens", f"{cost.get('output_tokens', 0):,}")


# ==================== PAGE 4: Methodology ====================
elif page == "Methodology":
    st.title("Methodology & Statistical Rigor")

    st.markdown("""
### Evaluation Design

- **Identical test sets**: All models evaluated on the exact same test split (`seed=42`)
- **German Sentiment**: Fixed HuggingFace split (1,490 test samples)
- **Yelp Reviews**: 5,000 stratified test samples (2K neg, 1K neutral, 2K pos)
- **Label mapping**: Yelp 5-star ratings mapped to 3-class (1-2: negative, 3: neutral, 4-5: positive)

### Multi-Seed Evaluation
""")

    multi_seed = load_multi_seed()
    if multi_seed:
        agg = multi_seed.get("aggregated", {})
        st.markdown(f"**Seeds used**: {multi_seed.get('seeds', [])}")

        rows = []
        for metric, stats in agg.items():
            rows.append({
                "Metric": metric,
                "Mean": stats["mean"],
                "Std": stats["std"],
                "Values": str(stats["values"]),
            })
        if rows:
            st.dataframe(
                pd.DataFrame(rows).style.format({"Mean": "{:.4f}", "Std": "{:.6f}"}),
                use_container_width=True, hide_index=True,
            )
            st.info(
                "Std = 0.0000 on German Sentiment because the HuggingFace dataset has fixed splits. "
                "On Yelp, changing the seed varies the training subsample and would show actual variance."
            )
    else:
        st.info("Run `python run_training.py --model classical --multi-seed` to generate multi-seed results.")

    st.markdown("### Learning Curve")

    lc = load_learning_curve()
    if lc:
        lc_df = pd.DataFrame(lc)
        fig = px.line(
            lc_df, x="train_size", y="f1_weighted",
            markers=True,
            labels={"train_size": "Training Samples", "f1_weighted": "F1 (weighted)"},
            title="SVM Learning Curve (German Sentiment)",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            lc_df.style.format({"f1_weighted": "{:.4f}", "accuracy": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )
        st.success("F1 plateaus at ~5K samples, confirming that 10K is sufficient for training.")
    else:
        st.info("Run `python run_training.py --model classical --learning-curve` to generate learning curves.")

    st.markdown("""
### Cross-Dataset Comparison

Training on German shows BERT's larger advantage (+5.6pp over classical ML),
while on English Yelp the gap narrows (+1.8pp). This suggests BERT's contextual
embeddings provide more value for morphologically complex languages.
""")

    # Cross-dataset bar chart
    cross_data = [
        {"Dataset": "German", "Model": "BERT", "F1": 0.9119},
        {"Dataset": "German", "Model": "Best Classical (SVM)", "F1": 0.8562},
        {"Dataset": "Yelp", "Model": "BERT", "F1": 0.7607},
        {"Dataset": "Yelp", "Model": "Best Classical (LogReg)", "F1": 0.7431},
    ]
    fig = px.bar(
        pd.DataFrame(cross_data),
        x="Dataset", y="F1", color="Model",
        barmode="group",
        color_discrete_map={"BERT": "#10B981", "Best Classical (SVM)": "#3B82F6", "Best Classical (LogReg)": "#3B82F6"},
        title="BERT vs. Best Classical ML — Cross-Dataset",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE 5: Live Demo ====================
elif page == "Live Demo":
    st.title("Live Sentiment Demo")
    st.markdown("Enter German text to classify with the trained SVM model (TF-IDF).")

    # Check if model is available
    from pathlib import Path as _Path
    _model_path = _Path(__file__).parent / "models" / "classical_svm.joblib"
    _model_available = _model_path.exists()

    if not _model_available:
        st.warning(
            "No trained model found. To enable the live demo, train a model locally:\n\n"
            "```\npython run_training.py --model classical --classifier svm --dataset german\n```"
        )

    text_input = st.text_area(
        "Enter customer feedback:",
        height=100,
        placeholder="z.B. 'Das Produkt ist ausgezeichnet! Sehr gute Qualitaet.'",
    )

    if st.button("Classify", type="primary", disabled=not _model_available) and text_input.strip():
        with st.spinner("Classifying..."):
            try:
                from src.inference.predictor import SentimentPredictor

                predictor = SentimentPredictor.from_classical("classical_svm")
                result = predictor.predict_single(text_input)

                sentiment_colors = {
                    "negative": "🔴",
                    "neutral": "🟡",
                    "positive": "🟢",
                }
                emoji = sentiment_colors.get(result["label_name"], "⚪")

                st.markdown(f"### {emoji} {result['label_name'].capitalize()}")
                st.markdown(f"**Confidence:** {result['confidence']:.1%}")

                probs = result["probabilities"]
                fig = px.bar(
                    x=list(probs.values()),
                    y=list(probs.keys()),
                    orientation="h",
                    labels={"x": "Probability", "y": "Sentiment"},
                    color=list(probs.keys()),
                    color_discrete_map={
                        "negative": "#EF4444",
                        "neutral": "#F59E0B",
                        "positive": "#10B981",
                    },
                )
                fig.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


# ==================== PAGE 6: Batch Analysis ====================
elif page == "Batch Analysis":
    st.title("Batch Analysis")
    st.markdown("Upload a CSV file with a `text` column to classify all rows.")

    from pathlib import Path as _Path2
    _model_path2 = _Path2(__file__).parent / "models" / "classical_logistic_regression.joblib"

    if not _model_path2.exists():
        st.warning(
            "No trained model found. Train a model first to enable batch classification."
        )
        st.stop()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
            st.stop()

        st.dataframe(df.head(10))
        st.info(f"Found {len(df)} rows.")

        if st.button("Classify All", type="primary"):
            with st.spinner(f"Classifying {len(df)} texts..."):
                try:
                    from src.inference.predictor import SentimentPredictor

                    predictor = SentimentPredictor.from_classical()
                    results = predictor.predict_batch(df["text"].tolist())

                    df["sentiment"] = [r["label_name"] for r in results]
                    df["confidence"] = [r["confidence"] for r in results]

                    st.success("Classification complete!")
                    st.dataframe(df)

                    fig = px.histogram(
                        df, x="sentiment", color="sentiment",
                        color_discrete_map={
                            "negative": "#EF4444",
                            "neutral": "#F59E0B",
                            "positive": "#10B981",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "sentiment_results.csv",
                        "text/csv",
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
