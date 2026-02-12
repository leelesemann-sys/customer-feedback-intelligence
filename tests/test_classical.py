"""Tests for Classical ML models."""

import numpy as np
import pytest

from src.models.classical import ClassicalSentimentModel, CLASSIFIERS


def _make_model(clf_name: str = "logistic_regression") -> ClassicalSentimentModel:
    """Create a model configured for small test datasets (min_df=1)."""
    model = ClassicalSentimentModel(
        classifier_name=clf_name,
        tfidf_max_features=100,
        tfidf_ngram_range=(1, 1),
    )
    model.pipeline.named_steps["tfidf"].set_params(min_df=1)
    return model


class TestClassicalSentimentModel:
    @pytest.fixture
    def trained_model(self, sample_texts, sample_labels):
        model = _make_model("logistic_regression")
        model.train(sample_texts, sample_labels)
        return model

    def test_available_classifiers(self):
        assert "logistic_regression" in CLASSIFIERS
        assert "svm" in CLASSIFIERS
        assert "naive_bayes" in CLASSIFIERS

    def test_invalid_classifier_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            ClassicalSentimentModel(classifier_name="invalid")

    def test_train_returns_duration(self, sample_texts, sample_labels):
        model = _make_model()
        result = model.train(sample_texts, sample_labels)
        assert "train_duration_s" in result
        assert result["train_duration_s"] >= 0

    def test_train_with_validation(self, sample_texts, sample_labels):
        model = _make_model()
        result = model.train(
            sample_texts, sample_labels,
            val_texts=sample_texts, val_labels=sample_labels,
        )
        assert "val_f1_weighted" in result
        assert 0.0 <= result["val_f1_weighted"] <= 1.0

    def test_predict_shape(self, trained_model, sample_texts):
        preds = trained_model.predict(sample_texts)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(sample_texts)

    def test_predict_proba_shape(self, trained_model, sample_texts):
        proba = trained_model.predict_proba(sample_texts)
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (len(sample_texts), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_training_raises(self):
        model = _make_model()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(["test"])

    @pytest.mark.parametrize("clf_name", ["logistic_regression", "svm", "naive_bayes"])
    def test_all_classifiers_train_and_predict(self, clf_name, sample_texts, sample_labels):
        model = _make_model(clf_name)
        model.train(sample_texts, sample_labels)
        preds = model.predict(sample_texts)
        assert len(preds) == len(sample_texts)
        assert all(p in [0, 1, 2] for p in preds)

    def test_save_and_load(self, trained_model, sample_texts, tmp_path):
        save_path = str(tmp_path / "model.joblib")
        trained_model.save(save_path)

        loaded = ClassicalSentimentModel.load(save_path)
        preds_original = trained_model.predict(sample_texts)
        preds_loaded = loaded.predict(sample_texts)
        np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_model_name(self):
        model = _make_model("svm")
        assert model.name == "classical_svm"
