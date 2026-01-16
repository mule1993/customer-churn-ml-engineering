import pandas as pd
import joblib
import os
import mlflow
import mlflow.xgboost  # Specialized for XGBoost
import mlflow.sklearn  # To capture preprocessing and split logic
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Your custom imports
from src.customer_churn_ml_engineering.data.loader import load_training_data
from src.customer_churn_ml_engineering.features.preprocess import build_preprocessor
from src.customer_churn_ml_engineering.config import MODELS_DIR

def main():
    # --------------------------------------------------
    # 0️⃣ MLflow Remote Setup (The Bridge)
    # --------------------------------------------------
    # Replace with your EC2 Public IP or Domain
    REMOTE_TRACKING_URI = "http://98.91.74.78" 
    
    mlflow.set_tracking_uri(REMOTE_TRACKING_URI)
    
    # These credentials MUST match what you put in your Nginx .htpasswd file
    os.environ['MLFLOW_TRACKING_USERNAME'] = "admin" 
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "your_secret_password"
    
    mlflow.set_experiment("Customer_Churn_Production")
    
    # Specialized autologging for XGBoost
    mlflow.xgboost.autolog(log_models=True)

    

    # 1️⃣ Load data
    df = load_training_data()
    df = df.replace(" ", pd.NA)

    target_col = "Churn"
    X = df.drop(columns=[target_col, "customerID"])
    y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)

    # 2️⃣ Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3️⃣ Preprocessing
    preprocessor = build_preprocessor(X)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # 4️⃣ Start MLflow Run context
    with mlflow.start_run(run_name="XGBoost_Baseline_v1") as run:
        # Log custom tags for easier searching later
        mlflow.set_tag("developer", "Gemini_Partner")
        mlflow.set_tag("model_type", "XGBoost")

        # 5️⃣ Model training
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        # Autolog captures everything during this .fit() call
        model.fit(
            X_train_p, 
            y_train, 
            eval_set=[(X_test_p, y_test)], 
            verbose=False
        )

        # 6️⃣ Manual Logging (for artifacts not caught by autolog)
        # We manually log the preprocessor because it's a separate object
        mlflow.sklearn.log_model(sk_model=preprocessor, artifact_path="preprocessor")
        
        # 7️⃣ Local Save (Keeping your existing workflow)
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
        joblib.dump(model, MODELS_DIR / "model.joblib")

        print(f"✅ Training complete. Run ID: {run.info.run_id}")
        print(f"📈 View metrics by running: mlflow ui")

if __name__ == "__main__":
    main()
