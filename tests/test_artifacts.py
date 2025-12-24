from pathlib import Path


def test_model_artifacts_exist():
    models_dir = Path("models")

    assert (models_dir / "model.joblib").exists()
    assert (models_dir / "preprocessor.joblib").exists()
