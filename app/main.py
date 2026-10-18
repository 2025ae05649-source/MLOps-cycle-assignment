"""
FastAPI app that serves the heart disease model.

Endpoints:
  GET  /         - welcome message
  GET  /health   - health check (used by Kubernetes)
  POST /predict  - send patient data as JSON, get prediction + confidence
  GET  /metrics  - Prometheus metrics

Run locally:  uvicorn app.main:app --reload --port 8000
"""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

# prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API requests",
                        ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds",
                            "Request latency in seconds", ["endpoint"])
PREDICTION_COUNT = Counter("predictions_total", "Predictions by class",
                           ["prediction"])
app.mount("/metrics", make_asgi_app())

model = None


def get_model():
    global model
    if model is None:
        logger.info("Loading model from %s", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
    return model


class Patient(BaseModel):
    age: float = Field(..., ge=0, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=1, le=4, examples=[1])
    trestbps: float = Field(..., gt=0, examples=[145])
    chol: float = Field(..., gt=0, examples=[233])
    fbs: int = Field(..., ge=0, le=1, examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[2])
    thalach: float = Field(..., gt=0, examples=[150])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, examples=[2.3])
    slope: int = Field(..., ge=1, le=3, examples=[3])
    ca: int = Field(..., ge=0, le=3, examples=[0])
    thal: int = Field(..., examples=[6])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    path = request.url.path
    if not path.startswith("/metrics"):
        logger.info("%s %s -> %s (%.3fs)", request.method, path,
                    response.status_code, duration)
        REQUEST_COUNT.labels(request.method, path,
                             response.status_code).inc()
        REQUEST_LATENCY.labels(path).observe(duration)
    return response


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API. Go to /docs to try it."}


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok"}
    except Exception as e:
        logger.error("Model not loaded: %s", e)
        raise HTTPException(status_code=503, detail="model not loaded")


@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.model_dump()])
    m = get_model()
    pred = int(m.predict(df)[0])
    confidence = float(m.predict_proba(df)[0][pred])
    PREDICTION_COUNT.labels(str(pred)).inc()
    logger.info("input=%s prediction=%d confidence=%.3f",
                patient.model_dump(), pred, confidence)
    return {
        "prediction": pred,
        "label": "heart disease" if pred == 1 else "no heart disease",
        "confidence": round(confidence, 4),
    }
