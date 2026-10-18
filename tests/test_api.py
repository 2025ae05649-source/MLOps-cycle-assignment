import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
MODEL_FILE = os.path.join(BASE_DIR, "models", "model.pkl")
os.environ["MODEL_PATH"] = MODEL_FILE

from app.main import app  # noqa: E402

# API tests need the trained model, so skip them if train.py was not run
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL_FILE),
                                reason="model.pkl not found, run train.py")

client = TestClient(app)

with open(os.path.join(BASE_DIR, "sample_input.json")) as f:
    SAMPLE = json.load(f)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1


def test_predict_missing_field():
    bad = dict(SAMPLE)
    del bad["age"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_metrics_endpoint():
    client.post("/predict", json=SAMPLE)
    r = client.get("/metrics/")
    assert r.status_code == 200
    assert "predictions_total" in r.text
