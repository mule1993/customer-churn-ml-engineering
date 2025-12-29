import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.customer_churn_ml_engineering.data.loader import load_training_data
from src.customer_churn_ml_engineering.features.preprocess import build_preprocessor
from src.customer_churn_ml_engineering.models.evaluate import evaluate_model
from src.customer_churn_ml_engineering.config import TARGET_COL, TEST_SIZE, RANDOM_STATE, MODELS_DIR
from src.customer_churn_ml_engineering.utils.helpers import set_seed, ensure_dir
from pathlib import Path



def main():
    # --------------------------------------------------
    # 1️⃣ Load data
    # --------------------------------------------------
    df = load_training_data()

    # Defensive cleanup
    df = df.replace(" ", pd.NA)

    target_col = "Churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Normalize target
    y = y.map({"Yes": 1, "No": 0}).astype(int)

    # --------------------------------------------------
    # 2️⃣ Train / test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # 3️⃣ Preprocessing
    # --------------------------------------------------
    preprocessor = build_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # --------------------------------------------------
    # 4️⃣ Model training
    # --------------------------------------------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train_p, y_train)

    # --------------------------------------------------
    # 5️⃣ Save artifacts (INSIDE PROJECT)
    # --------------------------------------------------
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(model, MODELS_DIR / "model.joblib")

    print("✅ Training complete")
    print(f"📦 Artifacts saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
