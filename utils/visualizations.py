# utils/visualizations.py
"""
Visualization Module
Generates all charts and visual analytics for the BI application.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Color Palette ────────────────────────────────────────────────────────────
PALETTE = {
    "primary":    "#4F46E5",   # Indigo
    "secondary":  "#7C3AED",   # Violet
    "success":    "#10B981",   # Emerald
    "warning":    "#F59E0B",   # Amber
    "danger":     "#EF4444",   # Red
    "info":       "#3B82F6",   # Blue
    "knn":        "#4F46E5",
    "svm":        "#EC4899",
    "ann":        "#10B981",
    "bg":         "#0F172A",   # Dark background
    "card":       "#1E293B",
}

MODEL_COLORS = [PALETTE["knn"], PALETTE["svm"], PALETTE["ann"]]


def set_dark_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor':  '#1E293B',
        'figure.facecolor':'#0F172A',
        'text.color':      'white',
        'axes.labelcolor': 'white',
        'xtick.color':     'white',
        'ytick.color':     'white',
        'grid.color':      '#334155',
        'axes.edgecolor':  '#334155',
    })


# ─── Dataset Visualizations ───────────────────────────────────────────────────

def plot_target_distribution(df, target_col):
    """Plotly pie chart of target variable distribution."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, 'Count']
    fig = px.pie(
        counts, names=target_col, values='Count',
        title=f'Target Distribution — {target_col}',
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4,
    )
    fig.update_layout(paper_bgcolor='#0F172A', font_color='white',
                      title_font_size=16)
    return fig


def plot_correlation_heatmap(df):
    """Seaborn/Matplotlib correlation heatmap for numeric columns."""
    set_dark_style()
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr.columns)), max(6, len(corr.columns) - 1)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='coolwarm', center=0, ax=ax,
        linewidths=0.5, linecolor='#0F172A',
        cbar_kws={'shrink': 0.8},
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, pad=12)
    plt.tight_layout()
    return fig


def plot_feature_distributions(df, feature_cols, target_col=None):
    """Plotly histograms for numeric features."""
    numeric_feats = [c for c in feature_cols
                     if df[c].dtype in [np.float64, np.int64, float, int]][:6]
    if not numeric_feats:
        return None

    n = len(numeric_feats)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=numeric_feats)
    for i, feat in enumerate(numeric_feats):
        r, c = divmod(i, cols)
        if target_col and target_col in df.columns:
            for cat in df[target_col].unique():
                sub = df[df[target_col] == cat][feat].dropna()
                fig.add_trace(
                    go.Histogram(x=sub, name=str(cat), opacity=0.7,
                                 nbinsx=20, showlegend=(i == 0)),
                    row=r+1, col=c+1
                )
        else:
            fig.add_trace(
                go.Histogram(x=df[feat].dropna(), nbinsx=20,
                             marker_color=PALETTE["primary"]),
                row=r+1, col=c+1
            )

    fig.update_layout(
        title='Feature Distributions',
        height=280 * rows,
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font_color='white',
        barmode='overlay',
    )
    return fig


def plot_missing_values(df):
    """Bar chart of missing value counts per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None
    fig = px.bar(
        x=missing.index, y=missing.values,
        labels={'x': 'Column', 'y': 'Missing Count'},
        title='Missing Values per Column',
        color=missing.values,
        color_continuous_scale='Reds',
    )
    fig.update_layout(paper_bgcolor='#0F172A', font_color='white',
                      showlegend=False)
    return fig


# ─── Model Evaluation Visualizations ──────────────────────────────────────────

def plot_confusion_matrix(cm, class_labels, model_name):
    """Seaborn heatmap for confusion matrix."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues', ax=ax,
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'{model_name} — Confusion Matrix', fontsize=12)
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df):
    """Grouped bar chart comparing KNN, SVM, ANN across metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig = go.Figure()

    colors = MODEL_COLORS
    for i, row in comparison_df.iterrows():
        fig.add_trace(go.Bar(
            name=row['Model'],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors[i % len(colors)],
            text=[f"{row[m]:.1f}%" for m in metrics],
            textposition='outside',
        ))

    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis=dict(title='Score (%)', range=[0, 115]),
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font_color='white',
        legend=dict(bgcolor='#1E293B'),
        height=420,
    )
    return fig


def plot_accuracy_radar(comparison_df):
    """Radar/spider chart comparing models across metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig = go.Figure()

    colors = MODEL_COLORS
    for i, row in comparison_df.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]  # close the loop
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=row['Model'],
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.4,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 105]),
            bgcolor='#1E293B',
        ),
        paper_bgcolor='#0F172A',
        font_color='white',
        title='Model Performance Radar',
        height=400,
        legend=dict(bgcolor='#1E293B'),
    )
    return fig


def plot_ann_history(history):
    """Line chart for ANN training/validation accuracy & loss."""
    hist = history.history
    epochs = list(range(1, len(hist['accuracy']) + 1))

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Accuracy Over Epochs', 'Loss Over Epochs'])

    fig.add_trace(go.Scatter(x=epochs, y=hist['accuracy'],
                             name='Train Acc', line=dict(color=PALETTE["success"])),
                  row=1, col=1)
    if 'val_accuracy' in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist['val_accuracy'],
                                 name='Val Acc', line=dict(color=PALETTE["warning"],
                                                           dash='dash')),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=hist['loss'],
                             name='Train Loss', line=dict(color=PALETTE["danger"])),
                  row=1, col=2)
    if 'val_loss' in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist['val_loss'],
                                 name='Val Loss', line=dict(color=PALETTE["info"],
                                                            dash='dash')),
                      row=1, col=2)

    fig.update_layout(
        height=350,
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font_color='white',
        title='ANN Training History',
        legend=dict(bgcolor='#1E293B'),
    )
    return fig


def plot_feature_importance(importance_df):
    """Horizontal bar chart for feature importance."""
    fig = px.bar(
        importance_df.head(15),
        x='Importance', y='Feature',
        orientation='h',
        title='Feature Importance (Random Forest)',
        color='Importance',
        color_continuous_scale='Viridis',
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='#0F172A',
        font_color='white',
        showlegend=False,
        height=400,
    )
    return fig


def plot_knn_k_scores(k_scores):
    """Line chart showing KNN accuracy vs K value."""
    ks = list(k_scores.keys())
    accs = [v * 100 for v in k_scores.values()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=accs,
        mode='lines+markers',
        line=dict(color=PALETTE["knn"], width=2),
        marker=dict(size=7),
        name='Accuracy',
    ))
    best_k = max(k_scores, key=k_scores.get)
    fig.add_vline(x=best_k, line_dash='dash', line_color=PALETTE["warning"],
                  annotation_text=f'Best K={best_k}')

    fig.update_layout(
        title='KNN Accuracy vs K Value',
        xaxis_title='K',
        yaxis_title='Accuracy (%)',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font_color='white',
        height=350,
    )
    return fig
