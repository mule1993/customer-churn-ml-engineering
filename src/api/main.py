from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

from src.api.schemas import ChurnRequest, ChurnResponse
from src.customer_churn_ml_engineering.models.predict import predict_single
from src.customer_churn_ml_engineering.config import MODELS_DIR, PROJECT_ROOT
ARTIFACTS_DIR = MODELS_DIR

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("churn-api")
# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predict customer churn using a trained ML model"
)
import time
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"time={process_time:.3f}s"
    )

    return response
logger.info("🚀 Churn API started successfully")
#------------------------------
#health
#------------------------------
@app.get("/health", tags=["health"])
def health_check():
    """
    Health check endpoint used by:
    - Docker
    - Load balancers
    - Monitoring systems
    """
    logger.info("Health check endpoint called")
    return {
        "status": "ok",
        "service": "churn-api",
        "model_loaded": True
    }
# -----------------------------
# Load artifacts ONCE
# -----------------------------
try:
    MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
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
        logger.info(f"Prediction request received: {data}")
        # 1️⃣ Convert validated input → DataFrame
        input_df = pd.DataFrame([request.dict()])

        # 2️⃣ Run inference
        churn_proba, churn_pred = predict_single(
            df=input_df,
            model=model,
            preprocessor=preprocessor
        )
        logger.info(f"Prediction result: {churn_pred}")
        # 3️⃣ Return response
        return ChurnResponse(
            churn_probability=float(churn_proba),
            churn_prediction=int(churn_pred)
        )

    except Exception as e:
        # API-safe error (never expose stack traces)
        raise HTTPException(status_code=500, detail=str(e))
        logger.exception("Prediction failed")










