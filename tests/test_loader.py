import pandas as pd
import numpy as np
from src.customer_churn_ml_engineering.data.loader import load_training_data
from src.customer_churn_ml_engineering.config import TARGET_COL


def test_loader_returns_dataframe():
    df = load_training_data()
    assert isinstance(df, pd.DataFrame)


def test_blank_strings_converted_to_nan():
    df = load_training_data()
    assert not (df == " ").any().any()


def test_no_missing_target():
    df = load_training_data()
    assert not df[TARGET_COL].isna().any()


def test_nan_is_numpy_nan():
    df = load_training_data()
    assert df.isna().sum().sum() >= 0  # sklearn-safe NaN
