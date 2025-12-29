import subprocess
import sys
from pathlib import Path
import joblib


def test_train_creates_artifacts():
    """Training should create model and preprocessor artifacts"""

    result = subprocess.run(
        [sys.executable, "-m", "src.customer-churn-ml-engineering.models.train"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    model_path = Path("models/model.joblib")
    preprocessor_path = Path("models/preprocessor.joblib")

    assert model_path.exists()
    assert preprocessor_path.exists()

    # Artifacts must be loadable
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    assert model is not None
    assert preprocessor is not None

