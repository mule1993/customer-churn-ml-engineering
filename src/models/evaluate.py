from sklearn.metrics import roc_auc_score, classification_report


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model.
    """

    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        "roc_auc": roc_auc,
        "report": report,
    }
