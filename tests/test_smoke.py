import subprocess
import sys


def test_training_pipeline_runs():
    """Smoke test: training pipeline runs without crashing"""
    result = subprocess.run(
        [sys.executable, "-m", "src.customer_churn_ml_engineering.models.train"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr