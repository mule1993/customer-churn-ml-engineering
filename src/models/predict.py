# src/models/predict.py

from pathlib import Path
import joblib
import pandas as pd


# --------------------------------------------------
# Resolve project root safely
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def predict(df: pd.DataFrame):
    """
    Run inference on input dataframe.
    Assumes df matches training schema.
    """

    # --------------------------------------------------
    # Load artifacts
    # --------------------------------------------------
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "model.joblib")

    # --------------------------------------------------
    # Basic validation
    # --------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # --------------------------------------------------
    # Transform + predict
    # --------------------------------------------------
    X_processed = preprocessor.transform(df)
    preds = model.predict(X_processed)

    return preds
