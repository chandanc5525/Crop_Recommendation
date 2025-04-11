from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    logging.info(f"Accuracy: {acc}")
    logging.info(f"Classification Report:\n{report}")
    logging.info(f"Confusion Matrix:\n{matrix}")
