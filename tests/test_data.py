import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.download_data import clean  # noqa: E402
from src.preprocess import (FEATURES, build_preprocessor,  # noqa: E402
                            load_data, split_features_target)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "heart_clean.csv")


def make_raw_df():
    # small fake raw data that looks like the UCI file
    return pd.DataFrame({
        "age": [63.0, 67.0, 41.0], "sex": [1.0, 1.0, 0.0],
        "cp": [1.0, 4.0, 2.0], "trestbps": [145.0, 160.0, 130.0],
        "chol": [233.0, 286.0, 204.0], "fbs": [1.0, 0.0, 0.0],
        "restecg": [2.0, 2.0, 2.0], "thalach": [150.0, 108.0, 172.0],
        "exang": [0.0, 1.0, 0.0], "oldpeak": [2.3, 1.5, 1.4],
        "slope": [3.0, 2.0, 1.0], "ca": [0.0, np.nan, 0.0],
        "thal": [6.0, 3.0, np.nan], "num": [0, 2, 0],
    })


def test_clean_fills_missing_values():
    df = clean(make_raw_df())
    assert df.isnull().sum().sum() == 0


def test_clean_makes_binary_target():
    df = clean(make_raw_df())
    assert "num" not in df.columns
    assert sorted(df["target"].unique()) == [0, 1]
    assert df["target"].tolist() == [0, 1, 0]


def test_clean_data_file():
    df = load_data(DATA_PATH)
    assert df.shape[0] > 250
    assert df.isnull().sum().sum() == 0
    assert set(df["target"].unique()) == {0, 1}


def test_load_data_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_preprocessor_output_shape():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    assert list(X.columns) == FEATURES
    out = build_preprocessor().fit_transform(X)
    assert out.shape[0] == len(X)
    # one hot encoding adds columns, so it should be more than the inputs
    assert out.shape[1] > len(FEATURES)
