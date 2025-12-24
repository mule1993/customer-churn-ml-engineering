import pytest
from src.data.loader import load_training_data
from src.config import TARGET_COL

def test_loader_columns():
    df = load_training_data()
    assert TARGET_COL in df.columns
    assert df.shape[0] > 0
