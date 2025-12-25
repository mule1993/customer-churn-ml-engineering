# src/models/predict.py

from pathlib import Path
import joblib
import pandas as pd

from src.utils.schema import validate_schema

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
    #validate schema
    validate_schema(df)

    # --------------------------------------------------
    # Load artifacts
    # --------------------------------------------------
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "model.joblib")

    # --------------------------------------------------
    # Transform + predict
    # --------------------------------------------------
    X_processed = preprocessor.transform(df)
    preds = model.predict(X_processed)

    return preds
