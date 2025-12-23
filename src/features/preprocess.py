
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "Churn",
    id_col: str = "customerID",
):
    """
    Prepare features and target for model training.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    artifacts : dict
        Fitted preprocessing objects (encoders, scaler)
    """

    df = df.copy()

    # Target
    y = df[target_col].apply(lambda x: 1 if x == "Yes" else 0)

    # Drop ID and target
    X = df.drop(columns=[id_col, target_col])

    # Identify column types
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    encoders = {}

    # Encode categoricals
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }

    return X, y, artifacts
