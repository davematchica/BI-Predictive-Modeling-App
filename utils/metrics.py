# utils/metrics.py
"""
Metrics Utility Module
Extracts and formats evaluation metrics from classification reports.
"""

import pandas as pd
import numpy as np


def extract_metrics(report_dict):
    """
    Extract precision, recall, f1-score from a classification_report dict.
    Returns: (precision, recall, f1) as floats (weighted avg).
    """
    weighted = report_dict.get('weighted avg', {})
    precision = weighted.get('precision', 0.0)
    recall = weighted.get('recall', 0.0)
    f1 = weighted.get('f1-score', 0.0)
    return precision, recall, f1


def build_comparison_table(results_dict):
    """
    Build a comparison DataFrame from a dict of model results.
    results_dict: {'KNN': {...}, 'SVM': {...}, 'ANN': {...}}
    Each value must have 'accuracy' and 'report' keys.
    Returns a pandas DataFrame.
    """
    rows = []
    for model_name, result in results_dict.items():
        acc = result.get('accuracy', 0)
        precision, recall, f1 = extract_metrics(result.get('report', {}))
        rows.append({
            'Model': model_name,
            'Accuracy': round(acc * 100, 2),
            'Precision': round(precision * 100, 2),
            'Recall': round(recall * 100, 2),
            'F1-Score': round(f1 * 100, 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    return df


def get_best_model(comparison_df):
    """Return the name of the model with the highest accuracy."""
    return comparison_df.iloc[0]['Model']
