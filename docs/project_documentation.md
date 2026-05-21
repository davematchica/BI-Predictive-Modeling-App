# Project Documentation
## Business Intelligence Predictive Modeling Application for Student Performance Analysis

---

## 1. Introduction

This project presents a Business Intelligence (BI) web application designed to predict student academic performance using machine learning. The system integrates three classification algorithms — K-Nearest Neighbor (KNN), Support Vector Machine (SVM), and Artificial Neural Network (ANN) — into a unified Streamlit interface, enabling educators and administrators to make data-driven decisions.

The application follows the full predictive modeling lifecycle: from data collection and preprocessing through model training, evaluation, and interactive prediction.

---

## 2. Objectives

1. Develop a functional BI web application using Python and Streamlit.
2. Implement and compare KNN, SVM, and ANN classification algorithms.
3. Automate data preprocessing including missing value handling, encoding, and scaling.
4. Provide interactive visualizations for exploratory data analysis.
5. Enable real-time prediction of student outcomes through a user-friendly interface.
6. Support educational decision-making through predictive analytics.

---

## 3. Scope

**In Scope:**
- CSV and Excel dataset support
- Binary and multiclass classification
- KNN, SVM, ANN model training and evaluation
- Confusion matrix, classification report, accuracy, precision, recall, F1-score
- Interactive prediction interface
- Visual analytics dashboard

**Out of Scope:**
- Regression problems (numeric prediction)
- Real-time streaming data
- Database persistence between sessions
- Multi-user authentication (single-user application)

---

## 4. Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

| Phase | Activity |
|---|---|
| Business Understanding | Define the prediction objective (Pass/Fail classification) |
| Data Understanding | Explore dataset structure, distributions, and correlations |
| Data Preparation | Handle missing values, encode categoricals, scale features |
| Modeling | Train KNN, SVM, and ANN models |
| Evaluation | Compare models using accuracy, precision, recall, F1 |
| Deployment | Streamlit web interface for interactive use |

---

## 5. Dataset Description

### Built-in Sample Dataset
- **Records:** 80 student records
- **Features:** 12 input variables
- **Target:** `performance` (Pass / Fail)

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Student age (16–18) |
| gender | Categorical | Male / Female |
| study_hours | Numeric | Weekly study hours |
| attendance | Numeric | Attendance percentage |
| previous_score | Numeric | Prior assessment score |
| assignments_completed | Numeric | Assignments submitted (%) |
| parental_education | Categorical | High School / Bachelor / Master |
| internet_access | Categorical | Yes / No |
| extracurricular | Categorical | Yes / No |
| sleep_hours | Numeric | Average sleep hours |
| final_grade | Numeric | Final examination score |
| performance | Categorical | **Target** — Pass / Fail |

---

## 6. Data Preprocessing

### 6.1 Missing Value Handling
- **Numerical columns:** Imputed with column mean
- **Categorical columns:** Imputed with column mode
- Logged in the Preprocessing page for transparency

### 6.2 Categorical Encoding
- **Label Encoding** applied to all categorical features
- Encoder objects stored for inverse transformation during prediction

### 6.3 Feature Scaling
- **StandardScaler** applied to all feature columns
- Transforms data to mean=0, standard deviation=1
- Fitted on training data only; applied to test data to prevent data leakage

### 6.4 Train/Test Split
- Default: 80% train / 20% test
- Stratified splitting to preserve class distribution
- Configurable via the Preprocessing UI

---

## 7. Algorithm Discussion

### 7.1 K-Nearest Neighbor (KNN)

**Principle:** Classifies a new data point based on the majority class among its K nearest neighbors in the feature space.

**Key Parameters:**
- `K`: Number of neighbors (default: 5)
- `weights`: Uniform or distance-weighted voting
- `metric`: Distance function (Minkowski, Euclidean, Manhattan)

**Advantages:** Simple, no training phase, interpretable
**Disadvantages:** Slow prediction on large datasets, sensitive to irrelevant features

### 7.2 Support Vector Machine (SVM)

**Principle:** Finds a hyperplane in high-dimensional space that maximally separates classes. Kernel functions enable non-linear classification.

**Key Parameters:**
- `kernel`: RBF (default), Linear, Polynomial
- `C`: Regularization parameter (trade-off between margin and misclassification)
- `gamma`: Kernel coefficient for RBF and Poly

**Advantages:** Effective in high-dimensional spaces, robust to overfitting with proper C
**Disadvantages:** Computationally expensive for large datasets, requires feature scaling

### 7.3 Artificial Neural Network (ANN)

**Architecture:**
```
Input Layer → Dense(64, ReLU) → BatchNorm → Dropout(0.2)
           → Dense(32, ReLU) → Dropout(0.2)
           → Output Layer (Sigmoid for binary / Softmax for multiclass)
```

**Training:**
- Optimizer: Adam
- Loss: Binary Cross-Entropy (binary) / Sparse Categorical Cross-Entropy (multiclass)
- Epochs: Configurable (default 50)
- Batch Size: Configurable (default 16)

**Advantages:** Learns complex non-linear patterns, scalable to large datasets
**Disadvantages:** Requires more data, computationally intensive, less interpretable

---

## 8. Evaluation Metrics

| Metric | Formula | Description |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correct predictions |
| Precision | TP/(TP+FP) | Correctness of positive predictions |
| Recall | TP/(TP+FN) | Coverage of actual positives |
| F1-Score | 2×(P×R)/(P+R) | Harmonic mean of Precision & Recall |

**Confusion Matrix:** Visualizes true positives, false positives, true negatives, and false negatives for each class.

---

## 9. Results and Analysis

### Expected Performance (Sample Dataset)
Based on typical results with the included 80-record dataset:

| Model | Expected Accuracy |
|---|---|
| KNN (K=5) | ~87–92% |
| SVM (RBF) | ~90–95% |
| ANN (50 epochs) | ~88–93% |

*Actual results vary based on random seed, hyperparameters, and dataset characteristics.*

### Key Findings
- **Study hours**, **attendance**, and **previous score** are the strongest predictors of performance.
- SVM with RBF kernel generally achieves highest accuracy on balanced datasets.
- ANN benefits from larger datasets and more epochs.
- KNN's optimal K can be determined using the built-in K-search feature.

---

## 10. Conclusion

This Business Intelligence application successfully demonstrates:
1. A complete machine learning workflow from data ingestion to prediction
2. Effective comparison of KNN, SVM, and ANN algorithms
3. An accessible, interactive interface for non-technical users
4. Automated preprocessing pipelines that reduce manual data preparation effort

The system provides actionable insights for educational institutions to identify at-risk students early and allocate support resources effectively.

---

## 11. Recommendations

1. **Data Quality:** Collect more records (500+) for more reliable model training
2. **Feature Engineering:** Consider adding derived features (e.g., study efficiency index)
3. **Hyperparameter Tuning:** Use GridSearchCV for optimal model configuration
4. **Model Retraining:** Retrain periodically as new student data becomes available
5. **Threshold Optimization:** Adjust classification threshold based on intervention cost vs. false positive tolerance

---

## 12. Future Enhancements

| Enhancement | Priority | Description |
|---|---|---|
| Cross-Validation | High | K-fold CV for more reliable evaluation |
| ROC Curve | High | AUC-ROC visualization per model |
| SHAP Explainability | Medium | Feature contribution per prediction |
| Model Persistence | Medium | Save/load trained models (joblib/h5) |
| SQLite Integration | Medium | Store predictions and history |
| Hyperparameter Tuning | Medium | Automated grid search UI |
| Authentication | Low | Multi-user login system |
| Email Reports | Low | Auto-generate and email PDF reports |
| REST API | Low | Expose prediction endpoint via FastAPI |

---

*Document generated by BI Predictive Modeling Application — Intelligence Systems Final Project*
