from pathlib import Path

TARGET_COL = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def find_project_root(marker_files=("pyproject.toml", "setup.cfg", ".git")):
    """
    Walk up directories until a project marker is found.
    """
    path = Path(__file__).resolve()
    for parent in path.parents:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise RuntimeError("Project root not found")

# Project root (repo root)
PROJECT_ROOT = find_project_root()

# Standard directories
#ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"

# Ensure dirs exist (safe in prod & tests)
for d in [
    #DATA_DIR,
    #RAW_DATA_DIR,
    #PROCESSED_DATA_DIR,
    #ARTIFACTS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    #DATA_PATH,
]:
    d.mkdir(parents=True, exist_ok=True)
