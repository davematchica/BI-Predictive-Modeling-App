# preprocessing/feature_engineering.py
"""
Feature Engineering Module
Provides feature importance and selection utilities.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif


def compute_feature_importance(X, y, feature_cols):
    """
    Use a RandomForest to compute feature importances.
    Returns a sorted DataFrame of feature → importance.
    """
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame({'Feature': feature_cols,
                             'Importance': np.ones(len(feature_cols)) / len(feature_cols)})


def select_top_features(X, y, feature_cols, k=5):
    """Return names of top-k features using ANOVA F-test."""
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    return [feature_cols[i] for i, m in enumerate(mask) if m]
