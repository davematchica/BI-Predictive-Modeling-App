# models/svm_model.py
"""
Support Vector Machine (SVM) Model Module
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale', degree=3):
    """Train an SVM classifier and return the fitted model."""
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_svm(model, X_test, y_test):
    """
    Evaluate a trained SVM model.
    Returns dict with accuracy, report, confusion matrix, predictions.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        y_prob = None

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_prob,
    }
