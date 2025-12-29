# tests/test_predict.py

from pathlib import Path
import pandas as pd

from src.customer_churn_ml_engineering.models.predict import predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def test_predict_runs_and_returns_output():
    """
    Smoke-level inference test.
    Assumes train.py has already been run.
    """

    # --------------------------------------------------
    # 1️⃣ Assert artifacts exist
    # --------------------------------------------------
    assert (MODELS_DIR / "model.joblib").exists()
    assert (MODELS_DIR / "preprocessor.joblib").exists()

    # --------------------------------------------------
    # 2️⃣ Minimal valid input (MATCHES TRAIN SCHEMA)
    # --------------------------------------------------
    sample = pd.DataFrame([{
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

    # --------------------------------------------------
    # 3️⃣ Run inference
    # --------------------------------------------------
    preds = predict(sample)

    # --------------------------------------------------
    # 4️⃣ Assertions
    # --------------------------------------------------
    assert preds is not None
    assert len(preds) == 1
    assert preds[0] in (0, 1)
