# src/models/train.py

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.loader import load_training_data
from src.features.preprocessing import build_preprocessor


# --------------------------------------------------
# 🔒 Resolve project root safely (professional fix)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


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
