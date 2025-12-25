import logging
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.data.loader import load_training_data
from src.features.preprocess import build_preprocessor
from src.models.evaluate import evaluate_model
from src.config import TARGET_COL, TEST_SIZE, RANDOM_STATE, MODELS_DIR
from src.utils.helpers import set_seed, ensure_dir
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def split_data(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Encode target explicitly
    y = y.map({"No": 0, "Yes": 1})
    if y.isna().any():
      raise ValueError("Target contains invalid or missing values after encoding")

    assert not y.isna().any(), "Target contains NA"
    assert X.isna().sum().sum() >= 0  # allowed
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def train_model(X_train, y_train):
    print("X dtypes:\n", X_train.dtypes)
    print("y unique values:", y_train.unique())
    preprocessor = build_preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          objective="binary:logistic", eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_p, y_train)
    return model, preprocessor

def save_artifacts(model, preprocessor):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR.mkdir(exist_ok=True)
    ensure_dir(MODELS_DIR)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    logger.info("Artifacts saved successfully")

def main():
    set_seed(RANDOM_STATE)
    logger.info("Starting training pipeline")
    df = load_training_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model, preprocessor = train_model(X_train, y_train)
    evaluate_model(model, preprocessor, X_test, y_test)
    save_artifacts(model, preprocessor)
    logger.info("Training pipeline completed")

if __name__ == "__main__":
    main()
