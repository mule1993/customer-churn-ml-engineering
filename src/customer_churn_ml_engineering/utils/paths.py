from pathlib import Path

# Project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths (logical, not actual storage)
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
