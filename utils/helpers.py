# utils/helpers.py
"""
General Helper Utilities
"""

import pandas as pd
import numpy as np
import streamlit as st
import io


def dataframe_to_csv_bytes(df):
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')


def fig_to_bytes(fig):
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def display_metric_card(col, label, value, delta=None, icon="📊"):
    """Render a styled metric inside a Streamlit column."""
    with col:
        st.metric(label=f"{icon} {label}", value=value, delta=delta)


def render_workflow_step(step_num, title, description, status="pending"):
    """Render a workflow step card with status indicator."""
    icons = {"done": "✅", "active": "🔄", "pending": "⏳"}
    icon = icons.get(status, "⏳")
    colors = {"done": "#10B981", "active": "#F59E0B", "pending": "#64748B"}
    color = colors.get(status, "#64748B")

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px;
                    background:#1E293B; padding:12px 16px;
                    border-radius:8px; border-left:4px solid {color};
                    margin-bottom:8px;">
            <span style="font-size:20px;">{icon}</span>
            <div>
                <div style="font-weight:600; color:white; font-size:14px;">
                    Step {step_num}: {title}
                </div>
                <div style="color:#94A3B8; font-size:12px;">{description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_prediction_result(prediction, probability=None, class_labels=None):
    """Display a styled prediction result box."""
    # Safely resolve label — prediction may already be a string
    label = str(prediction)
    try:
        if class_labels is not None:
            idx = int(prediction)
            if idx < len(class_labels):
                label = str(class_labels[idx])
    except (ValueError, TypeError):
        label = str(prediction)

    conf_text = ""
    if probability is not None:
        try:
            max_prob = float(np.max(probability)) * 100
            conf_text = (
                f"<br><span style='font-size:16px; color:#94A3B8;'>"
                f"Confidence: {max_prob:.1f}%</span>"
            )
        except Exception:
            pass

    positive_keywords = ["pass", "graduate", "enrolled", "1", "high", "yes", "good"]
    color = "#10B981" if label.lower() in positive_keywords else "#EF4444"

    st.markdown(
        f"""
        <div style="text-align:center; padding:24px; background:#1E293B;
                    border-radius:12px; border:2px solid {color}; margin:16px 0;">
            <div style="font-size:14px; color:#94A3B8; margin-bottom:8px;">
                🎯 Predicted Outcome
            </div>
            <div style="font-size:36px; font-weight:700; color:{color};">
                {label}
            </div>
            {conf_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def progress_steps(steps, completed):
    """Render a linear progress indicator."""
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            done = i < completed
            color = "#10B981" if done else "#334155"
            st.markdown(
                f"""<div style="background:{color}; border-radius:6px;
                             padding:8px; text-align:center; font-size:11px;
                             color:{'white' if done else '#64748B'};">
                    {'✓ ' if done else ''}{step}
                </div>""",
                unsafe_allow_html=True,
            )


def is_numeric_column(df, col_name):
    """
    Robustly determine if a column should use a numeric input widget.
    Handles edge cases where dtype is object but values are numeric,
    and columns with very few unique values that are better as dropdowns.
    """
    col = df[col_name]

    # Explicitly non-numeric dtype → categorical
    if col.dtype == object:
        return False

    # Try converting to float — catches cases where int columns
    # have been read as object due to mixed data
    try:
        col.astype(float)
    except (ValueError, TypeError):
        return False

    # Numeric but only 2–10 unique values → treat as categorical dropdown
    # (e.g. binary flags stored as 0/1, or ordinal scores 1–5)
    n_unique = col.nunique()
    if n_unique <= 10:
        return False

    return True


def get_numeric_input_widget(col, col_name, df):
    """
    Generate a number input widget for a truly continuous numeric feature.
    Falls back to a text input if min/max/mean cannot be computed.
    """
    try:
        series = df[col_name].dropna()
        min_v  = float(series.min())
        max_v  = float(series.max())
        mean_v = float(series.mean())

        # Guard against degenerate case where min == max
        if min_v == max_v:
            max_v = min_v + 1.0

        return col.number_input(
            col_name,
            min_value=min_v,
            max_value=max_v,
            value=mean_v,
            key=f"pred_{col_name}",
        )
    except Exception:
        # Ultimate fallback — plain text box
        return col.text_input(col_name, value="0", key=f"pred_{col_name}")


def get_categorical_input_widget(col, col_name, df):
    """
    Generate a selectbox widget for a categorical feature column.
    Sorts options and handles NaN values gracefully.
    """
    try:
        options = sorted(
            [str(v) for v in df[col_name].dropna().unique().tolist()]
        )
        if not options:
            options = ["N/A"]
        return col.selectbox(col_name, options=options, key=f"pred_{col_name}")
    except Exception:
        return col.text_input(col_name, value="", key=f"pred_{col_name}")


def get_input_widget(col, col_name, df):
    """
    Master widget selector — automatically picks numeric or categorical
    based on the column's actual content, not just its dtype label.
    Use this instead of calling get_numeric/get_categorical directly.
    """
    if is_numeric_column(df, col_name):
        return get_numeric_input_widget(col, col_name, df)
    else:
        return get_categorical_input_widget(col, col_name, df)