from src.customer_churn_ml_engineering.data.loader import load_training_data
from src.customer_churn_ml_engineering.config import TARGET_COL


def test_target_is_binary_encoded():
    df = load_training_data()
    y = df[TARGET_COL].map({"No": 0, "Yes": 1})

    assert set(y.unique()).issubset({0, 1})
