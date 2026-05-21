# preprocessing/preprocessing.py
"""
Data Preprocessing Module
Handles missing values, encoding, scaling, and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st


def load_dataset(uploaded_file):
    """Load CSV or Excel dataset from uploaded file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def get_dataset_info(df):
    """Return a summary dictionary of dataset statistics."""
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
    }
    return info


def handle_missing_values(df, logs):
    """
    Impute missing values:
      - Numerical → mean
      - Categorical → mode
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col].fillna(df[col].mean(), inplace=True)
            logs.append(f"✅ '{col}': filled {missing} missing values with mean ({df[col].mean():.2f})")

    for col in categorical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logs.append(f"✅ '{col}': filled {missing} missing values with mode ('{mode_val}')")

    return df, logs


def encode_categorical(df, target_col, logs):
    """
    Encode categorical features using Label Encoding.
    Returns df, encoders dict, and updated logs.
    """
    encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logs.append(f"✅ '{col}': Label Encoded → classes: {list(le.classes_)}")

    return df, encoders, logs


def scale_features(X_train, X_test, logs):
    """Apply StandardScaler to training and test sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logs.append(f"✅ Features scaled using StandardScaler (mean≈0, std≈1)")
    return X_train_scaled, X_test_scaled, scaler, logs


def preprocess_pipeline(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline:
    1. Missing value handling
    2. Categorical encoding
    3. Train/test split
    4. Feature scaling
    Returns: X_train, X_test, y_train, y_test, scaler, encoders, logs
    """
    logs = []
    df = df.copy()

    # Step 1: Handle missing values
    df, logs = handle_missing_values(df, logs)

    # Step 2: Encode categoricals
    df, encoders, logs = encode_categorical(df, target_col, logs)

    # Step 3: Select features and target
    X = df[feature_cols].values
    y = df[target_col].values
    logs.append(f"✅ Features selected: {feature_cols}")
    logs.append(f"✅ Target column: '{target_col}'")

    # Step 4: Train/test split
    # Use stratify only when every class has at least 2 members
    class_counts = pd.Series(y).value_counts()
    can_stratify = (class_counts >= 2).all()
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if can_stratify else None
        )
        strat_note = "stratified" if can_stratify else "random (some classes have <2 samples)"
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        strat_note = "random (fallback)"
    logs.append(f"✅ Train/test split ({strat_note}): {1-test_size:.0%}/{test_size:.0%} "
                f"→ Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 5: Scale features
    X_train_sc, X_test_sc, scaler, logs = scale_features(X_train, X_test, logs)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, encoders, logs


def preprocess_single_input(input_dict, feature_cols, scaler, encoders):
    """
    Preprocess a single prediction input using fitted scaler and encoders.
    Returns a 2D numpy array ready for model prediction.
    """
    row = []
    for col in feature_cols:
        val = input_dict.get(col, 0)
        if col in encoders:
            try:
                val = encoders[col].transform([str(val)])[0]
            except Exception:
                val = 0
        row.append(float(val))

    arr = np.array(row).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled