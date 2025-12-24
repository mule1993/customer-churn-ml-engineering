import joblib
import pandas as pd


def load_artifacts(model_path, preprocess_path):
    model = joblib.load(model_path)
    artifacts = joblib.load(preprocess_path)
    return model, artifacts

def preprocess_for_inference(df, artifacts):
    df = df.copy()

    # Drop ID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode categoricals
    for col, encoder in artifacts["encoders"].items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    # Scale numerics
    df[artifacts["numeric_cols"]] = artifacts["scaler"].transform(
        df[artifacts["numeric_cols"]]
    )

    return df

def predict(df, model, artifacts):
    X = preprocess_for_inference(df, artifacts)
    return model.predict_proba(X)[:, 1]
