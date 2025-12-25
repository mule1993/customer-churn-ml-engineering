# src/models/predict.py

from pathlib import Path
import joblib
import pandas as pd
import sys
from src.utils.schema import validate_schema

# --------------------------------------------------
# Resolve project root safely
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def predict(df: pd.DataFrame):
    """
    Run inference on input dataframe.
    Assumes df matches training schema.
    """
    #validate schema
    validate_schema(df)

    # --------------------------------------------------
    # Load artifacts
    # --------------------------------------------------
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "model.joblib")

    # --------------------------------------------------
    # Transform + predict
    # --------------------------------------------------
    X_processed = preprocessor.transform(df)
    preds = model.predict(X_processed)
    
    return pd.Series(preds, name="prediction")


def run_batch(input_csv: Path, output_csv: Path) -> None:
    """
    Run batch inference from CSV to CSV.
    """
    df = pd.read_csv(input_csv)

    predictions = predict(df)

    output = df.copy()
    output["prediction"] = predictions

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    print(f"✅ Predictions written to {output_csv}")


def main():
    """
    CLI entry point:
    python -m src.models.predict input.csv output.csv
    """
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python -m src.models.predict <input_csv> <output_csv>"
        )

    input_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    run_batch(input_csv, output_csv)


if __name__ == "__main__":
    main()

