from sklearn.model_selection import train_test_split

from src.data.loader import load_data
from src.features.preprocess import build_preprocessor
from src.models.evaluate import evaluate_model

def train(config):
    df = load_data(config.data_path)

    X = df.drop(columns=[config.target])
    y = df[config.target]

    # ✅ Train / test split lives here
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    model = build_model(config)
    model.fit(X_train_proc, y_train)

    evaluate_model(model, X_test_proc, y_test)

    save_artifacts(model, preprocessor)

