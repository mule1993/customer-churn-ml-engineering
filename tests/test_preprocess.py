# tests/test_preprocess.py

import pandas as pd

from src.customer_churn_ml_engineering.features.preprocess import build_preprocessor


def test_preprocessor_excludes_target():
    """
    Ensure target column is NOT included in preprocessing.
    """

    df = pd.DataFrame({
        "customerID": ["A", "B"],
        "tenure": [1, 5],
        "MonthlyCharges": [50.0, 70.0],
        "Contract": ["Month-to-month", "Two year"],
        "Churn": [0, 1],
    })

    X = df.drop(columns=["Churn"])

    preprocessor = build_preprocessor(X)

    # 🔑 MUST FIT before inspecting transformers_
    preprocessor.fit(X)

    # Collect all columns used by the preprocessor
    used_columns = []
    for _, _, cols in preprocessor.transformers_:
        used_columns.extend(cols)

    assert "Churn" not in used_columns

