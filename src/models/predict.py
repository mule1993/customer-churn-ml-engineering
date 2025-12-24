import joblib
import pandas as pd
from src.config import MODELS_DIR

def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    return model, preprocessor

def predict(df: pd.DataFrame):
    model, preprocessor = load_artifacts()
    X_p = preprocessor.transform(df)
    return model.predict(X_p)
