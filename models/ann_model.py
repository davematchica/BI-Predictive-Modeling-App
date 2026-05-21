# models/ann_model.py
"""
Artificial Neural Network (ANN) Model Module using TensorFlow/Keras
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def remap_labels(y):
    """
    Remap arbitrary integer labels to a contiguous 0-based range.
    e.g. [39, 57, 98, ...] → [0, 1, 2, ...]
    Returns (y_remapped, fitted LabelEncoder).
    """
    le = LabelEncoder()
    y_new = le.fit_transform(y)
    return y_new, le


def build_ann(input_dim, num_classes, hidden_layers=(64, 32), dropout_rate=0.2):
    """
    Build and compile a Keras ANN model.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_ann(X_train, y_train, input_dim, num_classes,
              hidden_layers=(64, 32), dropout_rate=0.2,
              epochs=50, batch_size=16, validation_split=0.1):
    """
    Train ANN model.
    Automatically remaps labels to 0-based contiguous integers to avoid
    'label value outside valid range' errors.
    Returns: (model, history, label_encoder)
    """
    # Always remap so labels are 0, 1, 2, ... regardless of original values
    y_remapped, label_enc = remap_labels(y_train)
    true_num_classes = len(label_enc.classes_)

    model = build_ann(input_dim, true_num_classes, hidden_layers, dropout_rate)

    history = model.fit(
        X_train, y_remapped,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0
    )
    return model, history, label_enc


def evaluate_ann(model, X_test, y_test, num_classes, label_enc=None):
    """
    Evaluate a trained ANN model.
    label_enc: the LabelEncoder returned by train_ann (used to remap y_test).
    Returns dict with accuracy, report, confusion matrix, predictions.
    """
    raw_preds = model.predict(X_test, verbose=0)

    # Remap y_test to the same 0-based encoding used during training
    if label_enc is not None:
        try:
            y_test_mapped = label_enc.transform(y_test)
        except Exception:
            y_test_mapped = y_test
    else:
        y_test_mapped = y_test

    true_num_classes = len(label_enc.classes_) if label_enc is not None else num_classes

    if true_num_classes == 2:
        y_prob = raw_preds
        y_pred = (raw_preds > 0.5).astype(int).flatten()
    else:
        y_prob = raw_preds
        y_pred = np.argmax(raw_preds, axis=1)

    acc = accuracy_score(y_test_mapped, y_pred)
    report = classification_report(y_test_mapped, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_mapped, y_pred)

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_prob,
        "label_enc": label_enc,
    }