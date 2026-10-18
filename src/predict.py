"""
Step 4: Simple inference script using the saved model.

Run:  python src/predict.py
"""
import os

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# one sample patient (first row of the dataset)
SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def predict(patient, model=None):
    if model is None:
        model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([patient])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][pred])
    return pred, prob


if __name__ == "__main__":
    pred, prob = predict(SAMPLE)
    label = "Heart disease" if pred == 1 else "No heart disease"
    print(f"Prediction: {pred} ({label}), confidence: {prob:.2f}")
