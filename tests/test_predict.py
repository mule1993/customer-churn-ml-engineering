import pandas as pd
from src.models.predict import predict


def test_predict_returns_predictions():
    """Predict should return one prediction per row"""

    sample = pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [5, 20],
            "PhoneService": ["Yes", "Yes"],
            "MonthlyCharges": [70.5, 89.1],
            "TotalCharges": [350.0, 1780.0],
        }
    )

    preds = predict(sample)

    assert len(preds) == len(sample)
    assert set(preds).issubset({0, 1})
