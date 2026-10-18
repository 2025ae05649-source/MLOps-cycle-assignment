"""
Preprocessing helpers that are shared by training, the API and the tests.
Keeping them in one place makes sure the same steps are used everywhere.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# cp, restecg, slope, ca, thal have more than 2 values -> one hot encode
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
# sex, fbs, exang are already 0/1 so I just pass them through
BINARY_FEATURES = ["sex", "fbs", "exang"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET = "target"


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    return df


def split_features_target(df):
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )
    return preprocessor
