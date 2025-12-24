from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COL = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42
