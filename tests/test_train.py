import pytest
from src.data.loader import load_training_data
from src.models.train import split_data, train_model

def test_train_pipeline_runs():
    df = load_training_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model, preprocessor = train_model(X_train, y_train)
    X_train_p = preprocessor.transform(X_train)
    y_pred = model.predict(X_train_p)
    assert len(y_pred) == len(y_train)
