# 🧬 Biomarker Classification Pipeline

A comprehensive machine learning pipeline for breast cancer biomarker classification using multiple algorithms and data augmentation techniques to handle class imbalance.

## 📋 Overview

This project implements an end-to-end classification pipeline that:
- Addresses severe class imbalance (7.14:1 ratio)
- Compares multiple augmentation strategies
- Evaluates three different ML algorithms
- Achieves **98.89% ROC-AUC** with optimal configuration

## 🎯 Key Features

- **Data Augmentation Techniques**
  - Random Oversampling
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - ADASYN (Adaptive Synthetic Sampling)
  
- **Machine Learning Models**
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier

- **Comprehensive Evaluation**
  - ROC-AUC analysis
  - Confusion matrix
  - Cross-validation (5-fold stratified)
  - Feature importance analysis

## 📊 Dataset

- **Source**: Breast Cancer Wisconsin Dataset (scikit-learn)
- **Samples**: 569 patients
- **Features**: 30 biomarker measurements
- **Classes**: Malignant (0) vs Benign (1)
- **Artificial Imbalance**: 90:10 ratio created for realistic testing

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
```

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the complete pipeline:

```bash
python biomarker_classification.py
```

The script will:
1. Load and preprocess the dataset
2. Create artificial class imbalance
3. Apply multiple augmentation strategies
4. Train all model combinations
5. Generate visualizations
6. Save results to CSV

## 📈 Results

### Best Model Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.9889 |
| **Accuracy** | 0.9878 |
| **Recall** | 1.0000 |
| **Precision** | 0.9863 |
| **F1-Score** | 0.9931 |

**Winning Configuration**: Logistic Regression + Random Oversampling

### Model Comparison

All 12 model-augmentation combinations are evaluated and compared across:
- ROC-AUC scores
- Recall performance (critical for medical diagnosis)
- Overall accuracy metrics

## 📁 Output Files

- `biomarker_classification_results.png` - Comprehensive visualization dashboard
- `model_comparision_results.csv` - Complete results table

## 🔍 Visualization Dashboard

The pipeline generates a 6-panel visualization including:

1. **ROC-AUC Heatmap** - Model vs Augmentation comparison
2. **Recall Heatmap** - Sensitivity analysis across configurations
3. **Confusion Matrix** - Best model predictions
4. **ROC Curves** - Performance curves for all augmentation methods
5. **Feature Importance** - Top 15 biomarkers ranked by importance
6. **Performance Metrics** - Bar chart of final model scores

## 🧪 Cross-Validation

5-fold stratified cross-validation ensures robust performance:
- Mean ROC-AUC: **1.0000**
- Standard Deviation: **0.0000**

## 📝 Key Findings

1. **Logistic Regression** outperformed ensemble methods on this dataset
2. **Random Oversampling** and **ADASYN** tied for best augmentation strategy
3. Perfect recall (1.0) achieved across most configurations - critical for medical diagnosis
4. Feature importance analysis reveals key biomarkers for classification

## 🛠️ Customization

### Modify Class Imbalance Ratio

```python
# In the code, adjust the sampling size:
X_minority = X[y == 0].sample(n=50, random_state=42)  # Change n value
```

### Add New Models

```python
models = {
    'Your_Model': YourClassifier(params),
    # Add more models here
}
```

### Change Augmentation Parameters

```python
smote = SMOTE(random_state=42, k_neighbors=5)  # Adjust k_neighbors
adasyn = ADASYN(random_state=42, n_neighbors=5)  # Adjust n_neighbors
```

## 📚 Project Structure

```
├── biomarker_classification.py          # Main pipeline script
├── biomarker_classification_results.png # Visualization output
├── model_comparision_results.csv        # Results table
├── requirements.txt                     # Dependencies
└── README.md                           # This file
```

## ⚠️ Important Notes

- **Medical Context**: This is a demonstration project. Real medical applications require extensive validation and regulatory approval.
- **Class Imbalance**: The artificial 90:10 imbalance simulates real-world scenarios where certain conditions are rare.
- **Feature Scaling**: StandardScaler is applied to ensure fair comparison across features.

## 🔬 Technical Details

### Data Split
- **Training**: 80% (325 samples)
- **Testing**: 20% (82 samples)
- **Stratification**: Maintains class distribution in splits

### Evaluation Metrics
- **ROC-AUC**: Primary metric for imbalanced classification
- **Recall**: Prioritized for medical diagnosis (minimize false negatives)
- **Precision**: Balanced with recall for overall performance

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional augmentation techniques
- Deep learning models
- Hyperparameter optimization
- Real-time prediction interface

## 👤 Author

**Akanksha Waghamode**
- GitHub: https://github.com/akanksha3-3 
- LinkedIn: https://www.linkedin.com/in/akanksha-waghamode-25aa9724a/ 
- Email: akankshawaghamode2001@gmail.com

Created for biomarker classification research and education.

## 🙏 Acknowledgments

- Breast Cancer Wisconsin Dataset (Diagnostic) - UCI Machine Learning Repository
- scikit-learn and imbalanced-learn communities
- XGBoost development team

---
