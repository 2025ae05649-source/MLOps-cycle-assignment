import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import SAMPLE, predict  # noqa: E402
from src.preprocess import (build_preprocessor, load_data,  # noqa: E402
                            split_features_target)
from src.train import evaluate, get_models  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "heart_clean.csv")


def train_small_model():
    X, y = split_features_target(load_data(DATA_PATH))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)
    model = Pipeline([("preprocess", build_preprocessor()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_two_models_defined():
    models = get_models()
    assert "logistic_regression" in models
    assert "random_forest" in models


def test_model_better_than_random():
    model, X_test, y_test = train_small_model()
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.7


def test_evaluate_returns_all_metrics():
    model, X_test, y_test = train_small_model()
    metrics = evaluate(model, X_test, y_test)
    for key in ["test_accuracy", "test_precision", "test_recall",
                "test_roc_auc"]:
        assert key in metrics
        assert 0 <= metrics[key] <= 1


def test_predict_sample():
    model, _, _ = train_small_model()
    pred, prob = predict(SAMPLE, model=model)
    assert pred in (0, 1)
    assert 0.5 <= prob <= 1.0
