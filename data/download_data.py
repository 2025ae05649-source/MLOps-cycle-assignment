"""
Step 1: Download the Heart Disease dataset from UCI and clean it.

Uses the official UCI package `ucimlrepo` (dataset id = 45).

Run:  python data/download_data.py
Output:
    data/heart_raw.csv    - data as downloaded
    data/heart_clean.csv  - cleaned data used for training
"""
import os

import certifi

# python.org Python on Mac has no SSL certificates by default,
# so point it to certifi's certificates, otherwise the download fails
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from ucimlrepo import fetch_ucirepo  # noqa: E402

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(DATA_DIR, "heart_raw.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "heart_clean.csv")


def download():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets  # column "num", values 0-4
    df = X.join(y)
    df.to_csv(RAW_PATH, index=False)
    print("Downloaded", df.shape, "->", RAW_PATH)
    return df


def clean(df):
    df = df.copy()
    print("Missing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # only 'ca' and 'thal' have a few missing values (6 rows total)
    # they are categorical, so fill with the most common value
    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # original target is 0-4, assignment needs binary: 0 = no disease, 1 = disease
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    df = df.drop_duplicates()

    # these are categories stored as floats, make them ints
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        df[col] = df[col].astype(int)

    return df


if __name__ == "__main__":
    raw = download()
    cleaned = clean(raw)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print("Cleaned data saved", cleaned.shape, "->", CLEAN_PATH)
    print(cleaned["target"].value_counts())
