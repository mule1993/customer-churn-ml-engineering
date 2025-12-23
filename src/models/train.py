import joblib
from pathlib import Path

from xgboost import XGBClassifier


def train_model(X_train, y_train, output_dir: Path):
    """
    Train churn model and save artifact.

    Returns
    -------
    model : trained model
    """

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "churn_model.joblib"
    joblib.dump(model, model_path)

    return model

