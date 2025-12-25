# tests/test_predict.py

from pathlib import Path
import pandas as pd
import joblib

from src.models.predict import predict


# --------------------------------------------------
# Resolve project root the SAME WAY as train.py
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def test_predict_runs_and_returns_output():
    """
    Smoke-level inference test.
    Assumes train.py has already been run and artifacts exist.
    """

    # --------------------------------------------------
    # 1️⃣ Assert artifacts exist
    # --------------------------------------------------
    model_path = MODELS_DIR / "model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    assert model_path.exists(), "❌ model.joblib not found. Run train.py first."
    assert preprocessor_path.exists(), "❌ preprocessor.joblib not found. Run train.py first."

    # --------------------------------------------------
    # 2️⃣ Create minimal valid input
    # --------------------------------------------------
    sample = pd.DataFrame([{
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
    assert preds[0] in [0, 1]

