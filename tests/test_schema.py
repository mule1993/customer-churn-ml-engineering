# tests/test_schema.py

import pandas as pd
import pytest

from src.customer_churn_ml_engineering.utils.schema import validate_schema


def test_schema_missing_column():
    df = pd.DataFrame({"tenure": [1]})

    with pytest.raises(ValueError):
        validate_schema(df)
