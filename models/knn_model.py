# models/knn_model.py
"""
K-Nearest Neighbor (KNN) Model Module
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def train_knn(X_train, y_train, k=5, weights='uniform', metric='minkowski'):
    """Train a KNN classifier and return the fitted model."""
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        metric=metric
    )
    model.fit(X_train, y_train)
    return model


def evaluate_knn(model, X_test, y_test):
    """
    Evaluate a trained KNN model.
    Returns dict with accuracy, report, confusion matrix, predictions.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Probability scores if available
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


def find_optimal_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    """Try multiple K values and return accuracy scores."""
    scores = {}
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        scores[k] = acc
    return scores
