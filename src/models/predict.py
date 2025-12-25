import pandas as pd
import joblib
from pathlib import Path
from typing import Union

MODELS_DIR = Path("models")
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"

# 🔹 Load artifacts globally (once)
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

TARGET_ENCODING = {0: "No", 1: "Yes"}  # reverse mapping, if needed


def validate_input(df: pd.DataFrame):
    """Check that input has required columns and correct dtypes"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    expected_features = preprocessor.feature_names_in_
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Optional: ensure dtypes are compatible
    for col in expected_features:
        if pd.api.types.is_numeric_dtype(preprocessor._df[col].dtype):
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column {col} must be numeric")


def predict(df: pd.DataFrame) -> pd.Series:
    """
    Accepts raw dataframe, returns 0/1 predictions.
    Raises error if input is invalid.
    """
    # Validate input
    validate_input(df)

    # Apply preprocessing
    X_processed = preprocessor.transform(df)

    # Generate predictions
    preds = model.predict(X_processed)

    return pd.Series(preds, index=df.index)

def predict_labels(df: pd.DataFrame) -> pd.Series:
    preds = predict(df)
    return preds.map(TARGET_ENCODING)
