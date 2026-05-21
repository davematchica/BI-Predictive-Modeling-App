# app.py
"""
Business Intelligence Predictive Modeling Application
Student Performance Analysis Using KNN, SVM, and ANN
Author: Intelligence Systems Final Project
Run: streamlit run app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ── Local Modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing.preprocessing import (
    load_dataset, get_dataset_info, preprocess_pipeline, preprocess_single_input
)
from preprocessing.feature_engineering import compute_feature_importance
from models.knn_model import train_knn, evaluate_knn, find_optimal_k
from models.svm_model import train_svm, evaluate_svm
from models.ann_model import train_ann, evaluate_ann
from utils.metrics import build_comparison_table, get_best_model, extract_metrics
from utils.visualizations import (
    plot_target_distribution, plot_correlation_heatmap,
    plot_feature_distributions, plot_missing_values,
    plot_confusion_matrix, plot_model_comparison, plot_accuracy_radar,
    plot_ann_history, plot_feature_importance, plot_knn_k_scores
)
from utils.helpers import (
    render_workflow_step, show_prediction_result, progress_steps,
    dataframe_to_csv_bytes, get_numeric_input_widget, get_categorical_input_widget, get_input_widget
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BI Predictive Modeling App",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df_raw":         None,
        "df_processed":   None,
        "feature_cols":   [],
        "target_col":     None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "scaler":         None,
        "encoders":       {},
        "knn_model":      None,
        "svm_model":      None,
        "ann_model":      None,
        "ann_history":    None,
        "ann_label_enc":  None,
        "results":        {},
        "comparison_df":  None,
        "preprocessed":   False,
        "trained":        False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:16px 0 8px;'>
            <div style='font-size:36px;'>🎓</div>
            <div style='font-size:15px; font-weight:700; color:#E2E8F0;'>
                BI Predictive Modeling
            </div>
            <div style='font-size:11px; color:#94A3B8; margin-top:4px;'>
                Student Performance Analysis
            </div>
        </div>
        <hr style='border-color:#334155; margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard",
         "📂 Dataset",
         "⚙️ Preprocessing",
         "🤖 Model Training",
         "📊 Evaluation",
         "📈 Visual Analytics",
         "🔮 Prediction"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    # System status
    st.markdown("**System Status**")
    st.markdown(f"Dataset: {'✅ Loaded' if st.session_state.df_raw is not None else '❌ None'}")
    st.markdown(f"Preprocessing: {'✅ Done' if st.session_state.preprocessed else '❌ Pending'}")
    st.markdown(f"Models: {'✅ Trained' if st.session_state.trained else '❌ Pending'}")

    if st.session_state.comparison_df is not None:
        best = get_best_model(st.session_state.comparison_df)
        st.markdown(f"Best Model: 🏆 **{best}**")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#64748B; text-align:center;'>"
        "Intelligence Systems Final Project<br>© 2024</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
        <div class='hero-banner'>
            <p class='hero-title'>🎓 Business Intelligence Predictive Modeling Application</p>
            <p class='hero-subtitle'>
                Student Performance Analysis Using KNN · SVM · ANN<br>
                Intelligence Systems Final Project
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Overview cards
    df = st.session_state.df_raw
    c1, c2, c3, c4 = st.columns(4)
    rows = df.shape[0] if df is not None else 0
    cols = df.shape[1] if df is not None else 0
    models_trained = sum([
        st.session_state.knn_model is not None,
        st.session_state.svm_model is not None,
        st.session_state.ann_model is not None,
    ])
    best_acc = 0
    if st.session_state.comparison_df is not None:
        best_acc = st.session_state.comparison_df.iloc[0]["Accuracy"]

    for col, icon, label, val in [
        (c1, "📋", "Dataset Records", f"{rows:,}"),
        (c2, "🔢", "Features", str(cols)),
        (c3, "🤖", "Models Trained", f"{models_trained}/3"),
        (c4, "🏆", "Best Accuracy", f"{best_acc:.1f}%" if best_acc else "—"),
    ]:
        col.markdown(f"""
            <div class='bi-card' style='text-align:center;'>
                <div style='font-size:32px;'>{icon}</div>
                <div class='bi-card-title'>{label}</div>
                <div class='bi-card-value'>{val}</div>
            </div>
        """, unsafe_allow_html=True)

    # Workflow overview
    st.markdown("### 📋 Predictive Modeling Workflow")
    workflow = [
        ("Problem Identification", "Define the student performance prediction objective"),
        ("Data Collection",        "Upload CSV/Excel dataset"),
        ("Data Preprocessing",     "Handle missing values, encode, scale"),
        ("Feature Selection",      "Choose relevant input features"),
        ("Model Training",         "Train KNN, SVM, and ANN models"),
        ("Model Testing",          "Evaluate on hold-out test set"),
        ("Performance Evaluation", "Metrics: Accuracy, Precision, Recall, F1"),
        ("Prediction Output",      "Predict new student outcomes interactively"),
    ]
    completed = 0
    if df is not None:            completed = 2
    if st.session_state.preprocessed: completed = 4
    if st.session_state.trained:      completed = 7
    if st.session_state.trained:      completed = 8

    col_a, col_b = st.columns(2)
    for i, (title, desc) in enumerate(workflow):
        status = "done" if i < completed else ("active" if i == completed else "pending")
        with (col_a if i % 2 == 0 else col_b):
            render_workflow_step(i + 1, title, desc, status)

    # Model comparison summary
    if st.session_state.comparison_df is not None:
        st.markdown("### 🏆 Model Performance Summary")
        df_cmp = st.session_state.comparison_df
        best = df_cmp.iloc[0]
        st.markdown(
            f"<div class='best-model-banner'>🏆 Best Model: {best['Model']} — "
            f"Accuracy {best['Accuracy']:.2f}%</div>",
            unsafe_allow_html=True
        )
        st.dataframe(df_cmp.style.highlight_max(
            subset=['Accuracy','Precision','Recall','F1-Score'],
            color='#1d4e3a'
        ), use_container_width=True)

    # About section
    st.markdown("### ℹ️ About This Application")
    st.info("""
    This application demonstrates a complete **Business Intelligence Predictive Modeling** workflow
    for student academic performance analysis.

    **Algorithms Implemented:**
    - 🔵 **KNN (K-Nearest Neighbor)** — Distance-based classification with adjustable K
    - 🟣 **SVM (Support Vector Machine)** — Kernel-based classification (Linear, RBF, Poly)
    - 🟢 **ANN (Artificial Neural Network)** — Deep learning via TensorFlow/Keras

    **Get started:** Upload your dataset in the **📂 Dataset** page, then follow the navigation.
    """)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Dataset":
    st.markdown("## 📂 Dataset Management")

    tab_upload, tab_explore, tab_stats = st.tabs(
        ["📤 Upload Dataset", "🔍 Explore Data", "📊 Statistics"]
    )

    with tab_upload:
        st.markdown("### Upload Your Dataset")
        st.info("Supported formats: **CSV** (.csv) and **Excel** (.xlsx)")

        col_upload, col_sample = st.columns([2, 1])
        with col_upload:
            uploaded = st.file_uploader(
                "Drag & drop or click to upload",
                type=["csv", "xlsx"],
                help="Upload a student performance dataset"
            )
        with col_sample:
            st.markdown("**Or use sample dataset:**")
            sample_path = os.path.join(
                os.path.dirname(__file__), "dataset", "sample_dataset.csv"
            )
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Sample CSV",
                        data=f.read(),
                        file_name="sample_student_dataset.csv",
                        mime="text/csv",
                    )
                if st.button("📂 Load Sample Dataset"):
                    df = pd.read_csv(sample_path)
                    st.session_state.df_raw = df
                    st.success(f"✅ Sample dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns")

        if uploaded is not None:
            with st.spinner("Loading dataset..."):
                df = load_dataset(uploaded)
            if df is not None:
                st.session_state.df_raw = df
                st.success(f"✅ File uploaded — {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Explore ────────────────────────────────────────────────────────────────
    with tab_explore:
        if st.session_state.df_raw is None:
            st.warning("⚠️ Please upload a dataset first.")
        else:
            df = st.session_state.df_raw
            info = get_dataset_info(df)

            c1, c2, c3, c4 = st.columns(4)
            for col, icon, label, val in [
                (c1, "📋", "Rows",           f"{info['rows']:,}"),
                (c2, "🔢", "Columns",        str(info['columns'])),
                (c3, "❓", "Missing Values", str(info['missing_values'])),
                (c4, "🔁", "Duplicates",     str(info['duplicate_rows'])),
            ]:
                col.metric(f"{icon} {label}", val)

            st.markdown("### Dataset Preview")
            n_rows = st.slider("Rows to display", 5, min(100, len(df)), 10)
            st.dataframe(df.head(n_rows), use_container_width=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Numeric Columns:**")
                st.write(info['numeric_columns'] or ["None"])
            with col_r:
                st.markdown("**Categorical Columns:**")
                st.write(info['categorical_columns'] or ["None"])

            # Missing values chart
            fig_mv = plot_missing_values(df)
            if fig_mv:
                st.plotly_chart(fig_mv, use_container_width=True)
            else:
                st.success("✅ No missing values detected!")

    # ── Statistics ─────────────────────────────────────────────────────────────
    with tab_stats:
        if st.session_state.df_raw is None:
            st.warning("⚠️ Please upload a dataset first.")
        else:
            df = st.session_state.df_raw
            st.markdown("### Statistical Summary")
            st.dataframe(df.describe(include='all').T, use_container_width=True)

            st.markdown("### Column Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.notnull().sum().values,
                'Null': df.isnull().sum().values,
                'Unique': df.nunique().values,
            })
            st.dataframe(dtype_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Preprocessing":
    st.markdown("## ⚙️ Data Preprocessing")

    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_raw.copy()
    all_cols = df.columns.tolist()

    tab_config, tab_run = st.tabs(["🔧 Configuration", "▶️ Run Preprocessing"])

    with tab_config:
        st.markdown("### Feature & Target Selection")
        target_col = st.selectbox(
            "🎯 Target Column (variable to predict)",
            all_cols,
            index=len(all_cols) - 1,
            help="Select the column you want to predict"
        )

        remaining = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect(
            "📌 Feature Columns (input variables)",
            remaining,
            default=remaining,
            help="Select features to use for training"
        )

        st.markdown("### Preprocessing Options")
        col1, col2, col3 = st.columns(3)
        test_size = col1.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = col2.number_input("Random State", 0, 999, 42)
        col3.markdown("<br>", unsafe_allow_html=True)
        col3.info(f"Train: {1 - test_size:.0%} / Test: {test_size:.0%}")

        st.markdown("### Encoding & Scaling")
        st.info("""
        The pipeline applies automatically:
        - **Missing values:** Numeric → mean, Categorical → mode
        - **Encoding:** Label Encoding for all categorical columns
        - **Scaling:** StandardScaler (mean=0, std=1)
        """)

    with tab_run:
        if not feature_cols:
            st.error("Please select at least one feature column.")
            st.stop()

        if st.button("▶️ Run Preprocessing Pipeline", type="primary",
                     use_container_width=True):
            with st.spinner("Processing data..."):
                try:
                    results = preprocess_pipeline(
                        df, feature_cols, target_col,
                        test_size=test_size, random_state=int(random_state)
                    )
                    (X_train, X_test, y_train, y_test,
                     scaler, encoders, logs) = results

                    st.session_state.update({
                        "X_train": X_train, "X_test": X_test,
                        "y_train": y_train, "y_test": y_test,
                        "scaler": scaler, "encoders": encoders,
                        "feature_cols": feature_cols,
                        "target_col": target_col,
                        "preprocessed": True,
                    })
                    st.success("✅ Preprocessing complete!")

                    st.markdown("### 📋 Preprocessing Log")
                    for log in logs:
                        st.markdown(f"- {log}")

                    st.markdown("### 📐 Split Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Train Samples", len(X_train))
                    c2.metric("Test Samples",  len(X_test))
                    c3.metric("Features",      X_train.shape[1])

                    # Class distribution
                    st.markdown("### 🎯 Target Class Distribution")
                    classes, counts = np.unique(y_train, return_counts=True)
                    fig = px.bar(x=[str(c) for c in classes], y=counts,
                                 labels={'x': 'Class', 'y': 'Count'},
                                 title='Training Set Class Distribution',
                                 color=[str(c) for c in classes],
                                 color_discrete_sequence=px.colors.qualitative.Bold)
                    fig.update_layout(paper_bgcolor='#0F172A', font_color='white',
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}")

        elif st.session_state.preprocessed:
            st.success("✅ Preprocessing already complete. Ready to train models.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Train Samples", len(st.session_state.X_train))
            c2.metric("Test Samples",  len(st.session_state.X_test))
            c3.metric("Features",      st.session_state.X_train.shape[1])


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("## 🤖 Machine Learning Model Training")

    if not st.session_state.preprocessed:
        st.warning("⚠️ Please run preprocessing first.")
        st.stop()

    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test
    num_classes = len(np.unique(y_train))

    tab_knn, tab_svm, tab_ann, tab_all = st.tabs(
        ["🔵 KNN", "🟣 SVM", "🟢 ANN", "🚀 Train All"]
    )

    # ── KNN ───────────────────────────────────────────────────────────────────
    with tab_knn:
        st.markdown("### 🔵 K-Nearest Neighbor")
        st.markdown("""
        KNN classifies a sample by majority vote of its K nearest neighbors
        in the feature space. Simple, interpretable, and effective for many tasks.
        """)

        col1, col2, col3 = st.columns(3)
        k_value  = col1.slider("K (Neighbors)", 1, 25, 5)
        weights  = col2.selectbox("Weights", ["uniform", "distance"])
        metric   = col3.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"])

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("🔵 Train KNN", type="primary", use_container_width=True):
                with st.spinner("Training KNN..."):
                    model = train_knn(X_train, y_train, k=k_value,
                                      weights=weights, metric=metric)
                    result = evaluate_knn(model, X_test, y_test)
                    st.session_state.knn_model = model
                    st.session_state.results['KNN'] = result
                st.success(f"✅ KNN trained — Accuracy: {result['accuracy']*100:.2f}%")
                st.metric("Test Accuracy", f"{result['accuracy']*100:.2f}%")

        with col_b:
            if st.button("🔍 Find Optimal K", use_container_width=True):
                with st.spinner("Testing K values 1–20..."):
                    k_scores = find_optimal_k(X_train, y_train, X_test, y_test)
                    st.session_state['k_scores'] = k_scores
                best_k = max(k_scores, key=k_scores.get)
                st.success(f"Optimal K = {best_k} (Accuracy: {k_scores[best_k]*100:.2f}%)")

        if 'k_scores' in st.session_state:
            fig = plot_knn_k_scores(st.session_state.k_scores)
            st.plotly_chart(fig, use_container_width=True)

    # ── SVM ───────────────────────────────────────────────────────────────────
    with tab_svm:
        st.markdown("### 🟣 Support Vector Machine")
        st.markdown("""
        SVM finds the optimal hyperplane that maximizes the margin between classes.
        Kernel functions map data to higher-dimensional spaces for non-linear separation.
        """)

        col1, col2, col3 = st.columns(3)
        kernel = col1.selectbox("Kernel", ["rbf", "linear", "poly"])
        C      = col2.select_slider("C (Regularization)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
        gamma  = col3.selectbox("Gamma", ["scale", "auto"])

        if st.button("🟣 Train SVM", type="primary", use_container_width=True):
            with st.spinner(f"Training SVM ({kernel} kernel)..."):
                model = train_svm(X_train, y_train, kernel=kernel, C=float(C), gamma=gamma)
                result = evaluate_svm(model, X_test, y_test)
                st.session_state.svm_model = model
                st.session_state.results['SVM'] = result
            st.success(f"✅ SVM trained — Accuracy: {result['accuracy']*100:.2f}%")

            p, r, f1 = extract_metrics(result['report'])
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Accuracy",  f"{result['accuracy']*100:.2f}%")
            col_b.metric("Precision", f"{p*100:.2f}%")
            col_c.metric("Recall",    f"{r*100:.2f}%")
            col_d.metric("F1-Score",  f"{f1*100:.2f}%")

    # ── ANN ───────────────────────────────────────────────────────────────────
    with tab_ann:
        st.markdown("### 🟢 Artificial Neural Network (TensorFlow/Keras)")
        st.markdown("""
        ANN learns complex non-linear patterns via multiple layers of neurons.
        Uses ReLU activation, Adam optimizer, and dropout for regularization.
        """)

        col1, col2 = st.columns(2)
        layer1   = col1.slider("Hidden Layer 1 Units", 16, 256, 64)
        layer2   = col1.slider("Hidden Layer 2 Units", 8,  128, 32)
        layer3   = col1.slider("Hidden Layer 3 Units (0 = skip)", 0, 64, 0)
        epochs   = col2.slider("Epochs", 10, 200, 50)
        batch_sz = col2.select_slider("Batch Size", [8, 16, 32, 64], 16)
        dropout  = col2.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)

        hidden = (layer1, layer2) if layer3 == 0 else (layer1, layer2, layer3)

        if st.button("🟢 Train ANN", type="primary", use_container_width=True):
            with st.spinner(f"Training ANN ({epochs} epochs)..."):
                try:
                    model, history, label_enc = train_ann(
                        X_train, y_train,
                        input_dim=X_train.shape[1],
                        num_classes=num_classes,
                        hidden_layers=hidden,
                        dropout_rate=dropout,
                        epochs=epochs,
                        batch_size=batch_sz,
                    )
                    result = evaluate_ann(model, X_test, y_test, num_classes, label_enc)
                    st.session_state.ann_model = model
                    st.session_state.ann_history = history
                    st.session_state.ann_label_enc = label_enc
                    st.session_state.results['ANN'] = result
                    st.success(f"✅ ANN trained — Accuracy: {result['accuracy']*100:.2f}%")

                    fig = plot_ann_history(history)
                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    st.error("❌ TensorFlow not installed. Run: pip install tensorflow")

    # ── Train All ─────────────────────────────────────────────────────────────
    with tab_all:
        st.markdown("### 🚀 Train All Models")
        st.info("Train KNN, SVM, and ANN with default hyperparameters in one click.")

        if st.button("🚀 Train All Models Now", type="primary", use_container_width=True):
            results = {}

            # KNN
            with st.spinner("Training KNN..."):
                m = train_knn(X_train, y_train, k=5)
                r = evaluate_knn(m, X_test, y_test)
                st.session_state.knn_model = m
                results['KNN'] = r
            st.success(f"✅ KNN — {r['accuracy']*100:.2f}%")

            # SVM
            with st.spinner("Training SVM..."):
                m = train_svm(X_train, y_train)
                r = evaluate_svm(m, X_test, y_test)
                st.session_state.svm_model = m
                results['SVM'] = r
            st.success(f"✅ SVM — {r['accuracy']*100:.2f}%")

            # ANN
            try:
                with st.spinner("Training ANN (50 epochs)..."):
                    m, hist, label_enc = train_ann(X_train, y_train,
                                        input_dim=X_train.shape[1],
                                        num_classes=num_classes,
                                        epochs=50)
                    r = evaluate_ann(m, X_test, y_test, num_classes, label_enc)
                    st.session_state.ann_model = m
                    st.session_state.ann_history = hist
                    st.session_state.ann_label_enc = label_enc
                    results['ANN'] = r
                st.success(f"✅ ANN — {r['accuracy']*100:.2f}%")
            except ImportError:
                st.warning("⚠️ TensorFlow not available — ANN skipped.")

            st.session_state.results = results
            st.session_state.trained = True

            cmp = build_comparison_table(results)
            st.session_state.comparison_df = cmp
            best = get_best_model(cmp)

            st.markdown(
                f"<div class='best-model-banner'>🏆 Best Model: {best} — "
                f"Accuracy {cmp.iloc[0]['Accuracy']:.2f}%</div>",
                unsafe_allow_html=True
            )
            st.dataframe(cmp, use_container_width=True)

        elif st.session_state.comparison_df is not None:
            st.success("✅ All models already trained.")
            st.dataframe(st.session_state.comparison_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation":
    st.markdown("## 📊 Model Evaluation")

    if not st.session_state.results:
        st.warning("⚠️ Please train at least one model first.")
        st.stop()

    results    = st.session_state.results
    y_test     = st.session_state.y_test
    target_col = st.session_state.target_col

    # Class labels
    enc = st.session_state.encoders
    if target_col in enc:
        class_labels = list(enc[target_col].classes_)
    else:
        class_labels = [str(c) for c in np.unique(y_test)]

    # ── Individual model results ────────────────────────────────────────────
    tabs = st.tabs([f"{'🔵' if m=='KNN' else '🟣' if m=='SVM' else '🟢'} {m}"
                    for m in results] + ["📊 Comparison"])

    for tab, (model_name, result) in zip(tabs[:-1], results.items()):
        with tab:
            st.markdown(f"### {model_name} Results")

            p, r, f1 = extract_metrics(result['report'])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{result['accuracy']*100:.2f}%")
            c2.metric("Precision", f"{p*100:.2f}%")
            c3.metric("Recall",    f"{r*100:.2f}%")
            c4.metric("F1-Score",  f"{f1*100:.2f}%")

            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown("**Confusion Matrix**")
                fig = plot_confusion_matrix(
                    result['confusion_matrix'], class_labels, model_name
                )
                st.pyplot(fig)
                plt.close()

            with col_r:
                st.markdown("**Classification Report**")
                report_df = pd.DataFrame(result['report']).T
                st.dataframe(report_df.style.format(precision=3).background_gradient(
                    cmap='Blues', subset=['precision', 'recall', 'f1-score']
                ), use_container_width=True)

            if model_name == 'ANN' and st.session_state.ann_history is not None:
                st.markdown("**Training History**")
                fig = plot_ann_history(st.session_state.ann_history)
                st.plotly_chart(fig, use_container_width=True)

    # ── Comparison Tab ─────────────────────────────────────────────────────
    with tabs[-1]:
        st.markdown("### 📊 Model Performance Comparison")

        if len(results) < 2:
            st.info("Train more models to see comparison.")
        else:
            cmp = build_comparison_table(results)
            st.session_state.comparison_df = cmp
            best = get_best_model(cmp)

            st.markdown(
                f"<div class='best-model-banner'>🏆 Best Performing Model: "
                f"{best} — Accuracy {cmp.iloc[0]['Accuracy']:.2f}%</div>",
                unsafe_allow_html=True
            )
            st.dataframe(cmp.style.highlight_max(
                subset=['Accuracy','Precision','Recall','F1-Score'],
                color='#1d4e3a'
            ), use_container_width=True)

            col_l, col_r = st.columns(2)
            with col_l:
                fig = plot_model_comparison(cmp)
                st.plotly_chart(fig, use_container_width=True)
            with col_r:
                fig = plot_accuracy_radar(cmp)
                st.plotly_chart(fig, use_container_width=True)

            # Download comparison
            st.download_button(
                "⬇️ Download Comparison Report",
                data=dataframe_to_csv_bytes(cmp),
                file_name="model_comparison.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: VISUAL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visual Analytics":
    st.markdown("## 📈 Visual Analytics & Business Intelligence")

    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df         = st.session_state.df_raw
    target_col = st.session_state.target_col
    feat_cols  = st.session_state.feature_cols

    tab_dist, tab_corr, tab_feat, tab_target = st.tabs([
        "📊 Distributions", "🔗 Correlation", "⭐ Features", "🎯 Target"
    ])

    with tab_dist:
        st.markdown("### Feature Distributions")
        if feat_cols:
            fig = plot_feature_distributions(df, feat_cols, target_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric features to display.")
        else:
            fig = plot_feature_distributions(
                df, df.select_dtypes(include=[np.number]).columns.tolist()
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        st.markdown("### Correlation Heatmap")
        fig = plot_correlation_heatmap(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Need at least 2 numeric columns for correlation.")

    with tab_feat:
        st.markdown("### Feature Importance")
        if st.session_state.X_train is not None and feat_cols:
            imp_df = compute_feature_importance(
                st.session_state.X_train,
                st.session_state.y_train,
                feat_cols
            )
            fig = plot_feature_importance(imp_df)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Importance Table**")
            st.dataframe(imp_df.style.background_gradient(
                subset=['Importance'], cmap='viridis'
            ), use_container_width=True)
        else:
            st.info("Run preprocessing first to compute feature importance.")

    with tab_target:
        st.markdown("### Target Variable Analysis")
        if target_col and target_col in df.columns:
            fig = plot_target_distribution(df, target_col)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**{target_col} Value Counts**")
            vc = df[target_col].value_counts().reset_index()
            vc.columns = [target_col, 'Count']
            vc['Percentage'] = (vc['Count'] / len(df) * 100).round(2)
            st.dataframe(vc, use_container_width=True)
        else:
            st.info("Select a target column in the Preprocessing page first.")


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown("## 🔮 Student Performance Prediction")

    if not st.session_state.results:
        st.warning("⚠️ Please train at least one model first.")
        st.stop()

    df         = st.session_state.df_raw
    feat_cols  = st.session_state.feature_cols
    target_col = st.session_state.target_col
    scaler     = st.session_state.scaler
    encoders   = st.session_state.encoders

    # Class label lookup
    if target_col in encoders:
        class_labels = list(encoders[target_col].classes_)
    else:
        class_labels = None

    st.markdown("### Enter Student Data")
    st.info("Fill in the student's information below, then click **Predict**.")

    # Dynamic form — group features into rows of 3
    input_vals = {}
    num_cols = 3
    chunks = [feat_cols[i:i+num_cols] for i in range(0, len(feat_cols), num_cols)]

    for chunk in chunks:
        cols = st.columns(num_cols)
        for col, feat in zip(cols, chunk):
            input_vals[feat] = get_input_widget(col, feat, df)

    st.markdown("---")

    # Model selector
    available_models = {}
    if st.session_state.knn_model: available_models['🔵 KNN'] = ('KNN', st.session_state.knn_model)
    if st.session_state.svm_model: available_models['🟣 SVM'] = ('SVM', st.session_state.svm_model)
    if st.session_state.ann_model: available_models['🟢 ANN'] = ('ANN', st.session_state.ann_model)

    col_sel, col_btn = st.columns([2, 1])
    model_choice = col_sel.selectbox("Select Model for Prediction", list(available_models.keys()))
    predict_all  = col_sel.checkbox("Predict with All Models", value=True)

    if col_btn.button("🔮 Predict", type="primary", use_container_width=True):
        try:
            X_input = preprocess_single_input(input_vals, feat_cols, scaler, encoders)

            if predict_all:
                st.markdown("### Predictions from All Models")
                pred_cols = st.columns(len(available_models))

                for i, (label, (name, model)) in enumerate(available_models.items()):
                    with pred_cols[i]:
                        st.markdown(f"**{label}**")
                        if name == 'ANN':
                            ann_le = st.session_state.get('ann_label_enc')
                            raw = model.predict(X_input, verbose=0)
                            ann_classes = ann_le.classes_ if ann_le is not None else None
                            true_nc = len(ann_classes) if ann_classes is not None else 2
                            if true_nc == 2:
                                pred_idx = int((raw[0][0] > 0.5))
                                prob = np.array([[1 - raw[0][0], raw[0][0]]])
                            else:
                                pred_idx = int(np.argmax(raw, axis=1)[0])
                                prob = raw
                            # Map back to original label
                            if ann_le is not None:
                                orig_label = ann_le.inverse_transform([pred_idx])[0]
                                pred = int(orig_label) if str(orig_label).isdigit() else pred_idx
                            else:
                                pred = pred_idx
                        else:
                            pred = int(model.predict(X_input)[0])
                            try:
                                prob = model.predict_proba(X_input)
                            except:
                                prob = None

                        pred_label = class_labels[pred] if class_labels and pred < len(class_labels) else str(pred)
                        show_prediction_result(pred_label, prob, class_labels)

            else:
                name, model = available_models[model_choice]
                if name == 'ANN':
                    ann_le = st.session_state.get('ann_label_enc')
                    raw = model.predict(X_input, verbose=0)
                    ann_classes = ann_le.classes_ if ann_le is not None else None
                    true_nc = len(ann_classes) if ann_classes is not None else 2
                    if true_nc == 2:
                        pred_idx = int((raw[0][0] > 0.5))
                        prob = np.array([[1 - raw[0][0], raw[0][0]]])
                    else:
                        pred_idx = int(np.argmax(raw, axis=1)[0])
                        prob = raw
                    if ann_le is not None:
                        orig_label = ann_le.inverse_transform([pred_idx])[0]
                        pred = int(orig_label) if str(orig_label).isdigit() else pred_idx
                    else:
                        pred = pred_idx
                else:
                    pred = int(model.predict(X_input)[0])
                    try:
                        prob = model.predict_proba(X_input)
                    except:
                        prob = None

                pred_label = class_labels[pred] if class_labels and pred < len(class_labels) else str(pred)
                show_prediction_result(pred_label, prob, class_labels)

            # Probability bar chart
            if class_labels and prob is not None:
                st.markdown("### Probability Distribution")
                prob_arr = prob[0] if prob.ndim == 2 else [1 - prob[0], prob[0]]
                fig = px.bar(
                    x=class_labels[:len(prob_arr)],
                    y=[p * 100 for p in prob_arr],
                    labels={'x': 'Class', 'y': 'Probability (%)'},
                    color=class_labels[:len(prob_arr)],
                    color_discrete_sequence=['#EF4444', '#10B981'],
                    title='Class Probability Scores',
                )
                fig.update_layout(paper_bgcolor='#0F172A', font_color='white',
                                  showlegend=False, yaxis_range=[0, 110])
                st.plotly_chart(fig, use_container_width=True)

            # Export prediction
            pred_export = pd.DataFrame([input_vals])
            pred_export['Predicted'] = pred_label
            st.download_button(
                "⬇️ Export Prediction",
                data=dataframe_to_csv_bytes(pred_export),
                file_name="prediction_result.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")