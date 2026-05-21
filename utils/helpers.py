# utils/helpers.py
"""
General Helper Utilities
"""

import pandas as pd
import numpy as np
import streamlit as st
import io
import base64


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
    label = prediction
    if class_labels is not None and len(class_labels) > int(prediction):
        label = class_labels[int(prediction)]

    conf_text = ""
    if probability is not None:
        max_prob = float(np.max(probability)) * 100
        conf_text = f"<br><span style='font-size:16px; color:#94A3B8;'>Confidence: {max_prob:.1f}%</span>"

    color = "#10B981" if str(label).lower() in ["pass", "1", "high"] else "#EF4444"

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


def get_numeric_input_widget(col, col_name, df):
    """Generate a number input widget for a feature column."""
    min_v = float(df[col_name].min())
    max_v = float(df[col_name].max())
    mean_v = float(df[col_name].mean())
    return col.number_input(col_name, min_value=min_v, max_value=max_v,
                            value=mean_v, key=f"pred_{col_name}")


def get_categorical_input_widget(col, col_name, df):
    """Generate a selectbox widget for a categorical feature column."""
    options = df[col_name].unique().tolist()
    return col.selectbox(col_name, options=options, key=f"pred_{col_name}")
