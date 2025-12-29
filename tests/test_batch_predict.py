# tests/test_batch_predict.py

import pandas as pd
from pathlib import Path

from src.customer_churn_ml_engineering.models.predict import run_batch


def test_batch_inference(tmp_path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"

    df = pd.DataFrame([{
        "customerID": "0001-A",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 350.5
    }])

    df.to_csv(input_csv, index=False)

    run_batch(input_csv, output_csv)

    assert output_csv.exists()

    out = pd.read_csv(output_csv)
    assert "prediction" in out.columns
