"""
Streamlit Dashboard for Customer Feedback Intelligence.

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config import LABEL_NAMES, RESULTS_DIR


st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="📊",
    layout="wide",
)


def load_metrics() -> list[dict]:
    """Load all model metrics from results/metrics/."""
    metrics_dir = RESULTS_DIR / "metrics"
    if not metrics_dir.exists():
        return []
    results = []
    for f in sorted(metrics_dir.glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Live Demo", "Model Comparison", "Error Analysis", "Batch Analysis"],
)


# ==================== PAGE 1: Live Demo ====================
if page == "Live Demo":
    st.title("Live Sentiment Demo")
    st.markdown("Enter text to classify with any available model.")

    text_input = st.text_area(
        "Enter customer feedback:",
        height=100,
        placeholder="z.B. 'Das Produkt ist ausgezeichnet! Sehr gute Qualitaet.'",
    )

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model",
            ["classical", "llm"],
            help="Select which model to use for prediction.",
        )
    with col2:
        if model_type == "llm":
            deployment = st.selectbox("Deployment", ["gpt-4o-mini", "gpt-4o"])
            mode = st.selectbox("Mode", ["zero_shot", "few_shot"])

    if st.button("Classify", type="primary") and text_input.strip():
        with st.spinner("Classifying..."):
            try:
                from src.inference.predictor import SentimentPredictor

                if model_type == "classical":
                    predictor = SentimentPredictor.from_classical()
                elif model_type == "llm":
                    predictor = SentimentPredictor.from_llm(
                        deployment=deployment, mode=mode
                    )
                else:
                    st.error(f"Unknown model type: {model_type}")
                    st.stop()

                result = predictor.predict_single(text_input)

                # Display result
                sentiment_colors = {
                    "negative": "🔴",
                    "neutral": "🟡",
                    "positive": "🟢",
                }
                emoji = sentiment_colors.get(result["label_name"], "⚪")

                st.markdown(f"### {emoji} {result['label_name'].capitalize()}")
                st.markdown(f"**Confidence:** {result['confidence']:.1%}")

                # Probability bars
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

            except FileNotFoundError:
                st.error("Model not found. Please train the model first.")
            except Exception as e:
                st.error(f"Error: {e}")


# ==================== PAGE 2: Model Comparison ====================
elif page == "Model Comparison":
    st.title("Model Comparison")

    metrics = load_metrics()
    if not metrics:
        st.warning("No model metrics found. Train some models first!")
        st.stop()

    # Build comparison table
    rows = []
    for m in metrics:
        test = m.get("test", {})
        latency = m.get("latency", {})
        cost = m.get("cost", {})
        rows.append({
            "Model": m.get("model", "unknown"),
            "F1 (weighted)": test.get("f1_weighted", 0),
            "Accuracy": test.get("accuracy", 0),
            "F1 (macro)": test.get("f1_macro", 0),
            "Precision": test.get("precision_weighted", 0),
            "Recall": test.get("recall_weighted", 0),
            "Latency (ms)": latency.get("per_sample_ms", 0),
            "Cost/1K ($)": cost.get("cost_per_1k_predictions", 0) if cost else 0,
        })

    df = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)

    # Display table
    st.subheader("Performance Overview")
    st.dataframe(
        df.style.format({
            "F1 (weighted)": "{:.4f}",
            "Accuracy": "{:.4f}",
            "F1 (macro)": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "Latency (ms)": "{:.1f}",
            "Cost/1K ($)": "{:.4f}",
        }).highlight_max(
            subset=["F1 (weighted)", "Accuracy"], color="#d4edda"
        ).highlight_min(
            subset=["Latency (ms)"], color="#d4edda"
        ),
        use_container_width=True,
    )

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("F1 Score Comparison")
        fig = px.bar(
            df, x="Model", y="F1 (weighted)",
            color="F1 (weighted)",
            color_continuous_scale="Greens",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Latency Comparison")
        fig = px.bar(
            df, x="Model", y="Latency (ms)",
            color="Latency (ms)",
            color_continuous_scale="Reds_r",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(min(len(metrics), 3))
    for i, m in enumerate(metrics):
        cm = m.get("test", {}).get("confusion_matrix")
        if cm:
            with cols[i % 3]:
                st.markdown(f"**{m.get('model', 'unknown')}**")
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

    metrics = load_metrics()
    if not metrics:
        st.warning("No model metrics found.")
        st.stop()

    selected_model = st.selectbox(
        "Select model",
        [m.get("model", "unknown") for m in metrics],
    )

    model_data = next((m for m in metrics if m.get("model") == selected_model), None)
    if not model_data:
        st.error("Model data not found.")
        st.stop()

    test = model_data.get("test", {})
    report = model_data.get("test_classification_report", {})

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
                    "Support": report[label].get("support", 0),
                })
        if class_rows:
            class_df = pd.DataFrame(class_rows)
            st.dataframe(
                class_df.style.format({
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                }),
                use_container_width=True,
            )

    # Confusion matrix
    cm = test.get("confusion_matrix")
    if cm:
        st.subheader("Confusion Matrix")
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=LABEL_NAMES,
            y=LABEL_NAMES,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Cost info for LLM models
    cost = model_data.get("cost")
    if cost:
        st.subheader("API Cost")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost", f"${cost.get('total_cost_usd', 0):.4f}")
        c2.metric("Input Tokens", f"{cost.get('input_tokens', 0):,}")
        c3.metric("Output Tokens", f"{cost.get('output_tokens', 0):,}")


# ==================== PAGE 4: Batch Analysis ====================
elif page == "Batch Analysis":
    st.title("Batch Analysis")
    st.markdown("Upload a CSV file with a `text` column to classify all rows.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
            st.stop()

        st.dataframe(df.head(10))
        st.info(f"Found {len(df)} rows.")

        model_type = st.selectbox("Model", ["classical", "llm"])

        if st.button("Classify All", type="primary"):
            with st.spinner(f"Classifying {len(df)} texts..."):
                try:
                    from src.inference.predictor import SentimentPredictor

                    if model_type == "classical":
                        predictor = SentimentPredictor.from_classical()
                    else:
                        predictor = SentimentPredictor.from_llm()

                    results = predictor.predict_batch(df["text"].tolist())

                    df["sentiment"] = [r["label_name"] for r in results]
                    df["confidence"] = [r["confidence"] for r in results]

                    st.success("Classification complete!")
                    st.dataframe(df)

                    # Distribution
                    fig = px.histogram(df, x="sentiment", color="sentiment",
                                       color_discrete_map={
                                           "negative": "#EF4444",
                                           "neutral": "#F59E0B",
                                           "positive": "#10B981",
                                       })
                    st.plotly_chart(fig, use_container_width=True)

                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "sentiment_results.csv",
                        "text/csv",
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
