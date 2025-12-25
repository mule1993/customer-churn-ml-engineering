import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

def evaluate_model(model, preprocessor, X_test, y_test):
    X_test_p = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_p)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    for k, v in metrics.items():
        if k != "confusion_matrix":
            logger.info("%s: %.4f", k, v)
    logger.info("Confusion Matrix:\n%s", metrics["confusion_matrix"])
    return metrics
