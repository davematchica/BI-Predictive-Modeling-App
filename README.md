# 🎓 Business Intelligence Predictive Modeling Application
### Student Performance Analysis Using KNN, SVM, and ANN

---

## 📋 Project Overview

A complete Business Intelligence (BI) web application for predicting student academic performance using three machine learning algorithms: **K-Nearest Neighbor (KNN)**, **Support Vector Machine (SVM)**, and **Artificial Neural Network (ANN)**. Built with Python and Streamlit, this system demonstrates a full end-to-end predictive modeling workflow suitable for academic and professional use.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 Dataset Management | Upload CSV/Excel, preview, validate, and explore data |
| ⚙️ Preprocessing | Auto missing-value handling, encoding, and StandardScaler |
| 🔵 KNN | Adjustable K, distance metrics, optimal K search |
| 🟣 SVM | Kernel selection (RBF, Linear, Poly), C and gamma tuning |
| 🟢 ANN | Configurable layers, dropout, training history charts |
| 📊 Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 📈 Visual Analytics | Distributions, correlation heatmap, feature importance |
| 🔮 Prediction | Dynamic form, probability scores, export results |
| 📊 Comparison | Side-by-side bar charts and radar charts |

---

## 🛠️ Technologies

- **Frontend:** Streamlit
- **ML Framework:** Scikit-learn, TensorFlow/Keras
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **File Support:** CSV, Excel (.xlsx)

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourname/BI_Predictive_Modeling_App.git
cd BI_Predictive_Modeling_App

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python -m streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
BI_Predictive_Modeling_App/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── dataset/
│   └── sample_dataset.csv          # Built-in student dataset (80 records)
│
├── models/
│   ├── knn_model.py                # KNN training & evaluation
│   ├── svm_model.py                # SVM training & evaluation
│   └── ann_model.py                # ANN (TensorFlow/Keras) module
│
├── preprocessing/
│   ├── preprocessing.py            # Full preprocessing pipeline
│   └── feature_engineering.py     # Feature importance & selection
│
├── utils/
│   ├── visualizations.py           # All chart/plot functions
│   ├── metrics.py                  # Metrics extraction utilities
│   └── helpers.py                  # UI helper components
│
├── assets/
│   └── styles.css                  # Custom CSS styling
│
└── docs/
    └── project_documentation.md    # Full project documentation
```

---

## 📖 Usage Guide

### Step 1 — Upload Dataset
Navigate to **📂 Dataset** → Upload CSV or Excel file, or click "Load Sample Dataset".

### Step 2 — Preprocessing
Navigate to **⚙️ Preprocessing** → Select feature and target columns → Click **Run Preprocessing Pipeline**.

### Step 3 — Train Models
Navigate to **🤖 Model Training** → Train individual models or click **Train All Models Now**.

### Step 4 — Evaluate
Navigate to **📊 Evaluation** → View confusion matrices, classification reports, and model comparison charts.

### Step 5 — Predict
Navigate to **🔮 Prediction** → Fill in student data → Click **Predict** to see outcomes.

---

## 📊 Supported Datasets

The system is compatible with any classification dataset. Recommended student datasets:

| Dataset | Source | Link |
|---|---|---|
| Student Performance | UCI ML Repository | https://archive.ics.uci.edu/ml/datasets/student+performance |
| Student Dropout | Kaggle | https://www.kaggle.com/datasets |
| Academic Success | Kaggle | Search: "student academic performance" |

---

## 🧪 Algorithms

### 🔵 K-Nearest Neighbor (KNN)
- Non-parametric, distance-based classifier
- Parameters: K value, distance metric (Minkowski, Euclidean, Manhattan), weights
- Best for: Small-to-medium datasets, interpretable results

### 🟣 Support Vector Machine (SVM)
- Finds optimal hyperplane with maximum margin
- Kernels: RBF, Linear, Polynomial
- Best for: High-dimensional data, binary and multiclass classification

### 🟢 Artificial Neural Network (ANN)
- Multi-layer perceptron using TensorFlow/Keras
- Features: ReLU activation, Dropout, BatchNormalization, Adam optimizer
- Best for: Complex non-linear patterns, large datasets

---

## 📸 Screenshots

*[Dashboard Screenshot]*
*[Model Training Screenshot]*
*[Evaluation Screenshot]*
*[Prediction Interface Screenshot]*

---

## 👥 Contributors

| Name | Role |
|---|---|
| [Your Name] | Developer / Data Scientist |

---

## 📄 License

MIT License — free to use for academic and personal projects.

---

## 🙏 Acknowledgements

- Scikit-learn documentation
- TensorFlow/Keras documentation
- Streamlit community
- UCI Machine Learning Repository
