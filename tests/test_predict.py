import pytest
import pandas as pd
from src.models.predict import predict
from src.models.train import train_model
from src.features.preprocess import build_preprocessor
from src.config import MODELS_DIR
import joblib
import os

def setup_artifacts():
    if not (MODELS_DIR / "model.joblib").exists():
        os.makedirs(MODELS_DIR, exist_ok=True)
        df = pd.DataFrame({
            "tenure": [1,2],
            "MonthlyCharges": [20,30],
            "TotalCharges": [20,30],
            "gender": ["Male","Female"],
            "Partner": ["Yes","No"],
            "Dependents": ["No","Yes"],
            "PhoneService": ["Yes","No"],
            "InternetService": ["DSL","Fiber optic"],
            "Contract": ["Month-to-month","One year"],
            "PaymentMethod": ["Electronic check","Mailed check"],
        })
        preprocessor = build_preprocessor()
        X_p = preprocessor.fit_transform(df)
        model, _ = train_model(df.drop(columns=["Churn"], errors='ignore'), df.get("Churn", [0,1]))
        joblib.dump(model, MODELS_DIR / "model.joblib")
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

def test_predict_runs():
    setup_artifacts()
    df = pd.DataFrame({
        "tenure": [5],
        "MonthlyCharges": [50],
        "TotalCharges": [100],
        "gender": ["Male"],
        "Partner": ["No"],
        "Dependents": ["No"],
        "PhoneService": ["Yes"],
        "InternetService": ["DSL"],
        "Contract": ["Month-to-month"],
        "PaymentMethod": ["Electronic check"],
    })
    preds = predict(df)
    assert len(preds) == 1
