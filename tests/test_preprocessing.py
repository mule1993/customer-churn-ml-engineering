import pytest
import pandas as pd
from src.features.preprocess import build_preprocessor

def sample_dataframe():
    return pd.DataFrame({
        "tenure": [1,2],
        "MonthlyCharges": [20,30],
        "TotalCharges": [20,30],
        "gender": ["Male","Female"],
        "Partner": ["Yes","No"],
        "Dependents": ["No","Yes"],
        "PhoneService": ["Yes","No"],
        "InternetService": ["DSL","Fiber optic"],
        "Contract": ["Month-to-month","One year"],
        "PaymentMethod": ["Electronic check","Mailed check"],
    })

def test_preprocessor_output_shape():
    df = sample_dataframe()
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == df.shape[0]
    assert transformed.shape[1] > 0
