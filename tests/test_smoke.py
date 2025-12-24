import pytest
import pandas as pd
from src.models.train import train_model, split_data
from src.features.preprocess import build_preprocessor
from src.models.evaluate import evaluate_model
from src.models.predict import predict
from src.config import TARGET_COL

def smoke_sample_df():
    return pd.DataFrame({
        "tenure": [1, 2],
        "MonthlyCharges": [20, 30],
        "TotalCharges": [20, 30],
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "No"],
        "InternetService": ["DSL", "Fiber optic"],
        "Contract": ["Month-to-month", "One year"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        TARGET_COL: [0, 1],
    })

def test_pipeline_smoke():
    df = smoke_sample_df()
    X_train, X_test, y_train, y_test = split_data(df)
    model, preprocessor = train_model(X_train, y_train)
    metrics = evaluate_model(model, preprocessor, X_test, y_test)
    preds = predict(X_test)
    assert len(preds) == len(y_test)
    assert "accuracy" in metrics
