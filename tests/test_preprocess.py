from src.data.loader import load_training_data
from src.features.preprocess import build_preprocessor
from src.config import TARGET_COL


def test_preprocessor_excludes_target():
    df = load_training_data()
    X = df.drop(columns=[TARGET_COL])

    preprocessor = build_preprocessor(X)

    # Target must NOT be among features
    all_features = (
        preprocessor.transformers_[0][2].tolist()
        + preprocessor.transformers_[1][2].tolist()
    )

    assert TARGET_COL not in all_features

