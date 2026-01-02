from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

from src.api.schemas import ChurnRequest, ChurnResponse
from src.customer_churn_ml_engineering.models.predict import predict_single
from src.customer_churn_ml_engineering.config import MODELS_DIR
ARTIFACTS_DIR = MODELS_DIR
# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predict customer churn using a trained ML model"
)

# -----------------------------
# Load artifacts ONCE
# -----------------------------
try:
    MODELS_DIR = PROJECT_ROOT / "artifacts" / "models
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    try:
        # 1️⃣ Convert validated input → DataFrame
        input_df = pd.DataFrame([request.dict()])

        # 2️⃣ Run inference
        churn_proba, churn_pred = predict_single(
            df=input_df,
            model=model,
            preprocessor=preprocessor
        )

        # 3️⃣ Return response
        return ChurnResponse(
            churn_probability=float(churn_proba),
            churn_prediction=int(churn_pred)
        )

    except Exception as e:
        # API-safe error (never expose stack traces)
        raise HTTPException(status_code=500, detail=str(e))

