#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12,6)


# DATA LOADING

print("\nLoading Dataset...")
print("-"*80)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='diagnosis')

X.info()
print(f"Dataset shape: {X.shape}")
print(f"Number of Features: {X.shape[1]}")
print(f"Number of Samples: {X.shape[0]}")
print(f"\nTarget Distribution:")
print(y.value_counts())
print(f"\nClass Balance: {y.value_counts(normalize=True).to_dict()}")

print(f"\nFeature Statistics:")
print(X.describe().iloc[:, :5])

print(f"\nMissing Values: {X.isnull().sum().sum()}")


# DATA PREPROCESSING

print("\nData Preprocessing...")
print("-"*80)

print("\nCreating artificial class imbalance (90:10)...")
X_majority = X[y == 1]
y_majority = y[y == 1]
X_minority = X[y == 0].sample(n=50, random_state=42)
y_minority = y[y == 0].loc[X_minority.index]

X_imbalanced = pd.concat([X_majority, X_minority])
y_imbalanced = pd.concat([y_majority, y_minority])

print(f"\nNew Class Distribution:")
print(y_imbalanced.value_counts())
print(f"Imbalance Ratio: {y_imbalanced.value_counts()[1]/y_imbalanced.value_counts()[0]:.2f}:1")

print("\nSplitting data: 80% train, 20% test (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imbalanced, y_imbalanced,
    test_size=0.2,
    random_state=42,
    stratify=y_imbalanced
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training class distribution:\n{y_train.value_counts()}")

print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
print("\nStandardScaler Applied.")


# HNADLING CLASS IMBALANCE

print("\nHandling Class Imbalance...")
print("-"*80)

augmentation_strategies = {}

print("\n1. Baseline (No Augmentation)")
augmentation_strategies['Baseline'] = (X_train_scaled.values, y_train.values)

print("2. Random Oversampling...")
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train_scaled, y_train)
augmentation_strategies['Random_Oversampling'] = (X_ros, y_ros)
print(f"  After ROS: {pd.Series(y_ros).value_counts().to_dict()}")

print("3. SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
augmentation_strategies['SMOTE'] = (X_smote, y_smote)
print(f"  After SMOTE: {pd.Series(y_smote).value_counts().to_dict()}")

print("4. ADASYN...")
adasyn = ADASYN(random_state=42, n_neighbors=5)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train_scaled, y_train)
augmentation_strategies['ADASYN'] = (X_adasyn, y_adasyn)
print(f"  After ADASYN: {pd.Series(y_adasyn).value_counts().to_dict()}")


# MODEL TRAINING

print("\nTraining Multiple Models...")
print("-"*80)

models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = []

for aug_name, (X_aug, y_aug) in augmentation_strategies.items():
    print(f"\n--- Training With {aug_name} ---")

    for model_name, model in models.items():
        print(f"  Training {model_name}...", end=" ")

        model.fit(X_aug, y_aug)

        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results.append({
            'Augmentation': aug_name,
            'Model': model_name,
            'Accuracy':accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'ROC-AUC': roc_auc
        })
        
        print(f"ROC-AUC: {roc_auc:.3f}")


# MODEL EVALUATION

print("\nModel Evaluation...")
print("-"*80)

results_df = pd.DataFrame(results)
print("\nComplete Results Table:")
print(results_df.to_string(index=False))

best_idx = results_df['ROC-AUC'].idxmax()
best_result = results_df.loc[best_idx]
print(f"\n{'='*80}")
print(f"BEST MODEL: {best_result['Model']} with {best_result['Augmentation']}")
print(f"ROC-AUC: {best_result['ROC-AUC']:.4f}")
print(f"Accuracy: {best_result['Accuracy']:.4f}")
print(f"Recall: {best_result['Recall']:.4f}")
print(f"{'='*80}")

best_aug = best_result['Augmentation']
best_model_name = best_result['Model']

X_best, y_best = augmentation_strategies[best_aug]
best_model = models[best_model_name]
best_model.fit(X_best, y_best)

y_pred_best = best_model.predict(X_test_scaled)
y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Malignant', 'Benign']))

print("\nConfucion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
print(f"\nTrue Negative: {cm[0,0]}")
print(f"\nFalse Negative: {cm[0,1]}")
print(f"\nTrue Positive: {cm[1,0]}")
print(f"\nFalse Positive: {cm[1,1]}")


# FEATURE IMPORTANCE

print("\nFeature Importance Analysis...")
print("-"*80)

if best_model_name in ['Random_Forest', 'XGBoost']:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    top_features = feature_importance.head(15)['Feature'].tolist()

print("done.")


# VISUALISATION

print("\nGenerating Visualizations...")
print("-"*80)

fig = plt.figure(figsize=(16, 12))

# 1. ROC_AUC Comparision
ax1 = plt.subplot(2,3,1)
pivot_data = results_df.pivot(index='Model', columns='Augmentation', values='ROC-AUC')
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax1)
ax1.set_title('ROC-AUC Comparision', fontsize=12, fontweight='bold')

# 2. Recall Comparision
ax2 = plt.subplot(2,3,2)
pivot_recall = results_df.pivot(index='Model', columns='Augmentation', values='Recall')
sns.heatmap(pivot_recall, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
ax2.set_title('Recall Comparision', fontsize=12, fontweight='bold')

# 3. Confusion Matrix
ax3 = plt.subplot(2,3,3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
           xticklabels=['Malignant', 'Benign'],
           yticklabels=['Malignant', 'Benign'])
ax3.set_title(f'Confusion Matrix\n({best_model_name} + {best_aug})',
             fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# 4. ROC Curves
ax4 = plt.subplot(2,3,4)
for aug_name, (X_aug, y_aug) in augmentation_strategies.items():
    model = models[best_model_name]
    model.fit(X_aug, y_aug)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax4.plot(fpr, tpr, label=f'{aug_name} (AUC={auc:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', label='Random Classfier')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title(f'ROC Curves - {best_model_name}', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right', fontsize=8)
ax4.grid(True, alpha=0.3)

# 5: Feature Importance Bar Chart
ax5 = plt.subplot(2, 3, 5)
if best_model_name in ['Random_Forest', 'XGBoost']:
    top_15 = feature_importance.head(15) 
    
    ax5.barh(range(len(top_15)), top_15['Importance'], color='steelblue')
    ax5.set_yticks(range(len(top_15)))
    ax5.set_yticklabels(top_15['Feature'], fontsize=8)
    ax5.set_xlabel('Importance Score')
    ax5.set_title('Top 15 Features by Importance', fontsize=12, fontweight='bold')
    ax5.invert_yaxis() 
else:
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': np.abs(best_model.coef_[0]) 
    }).sort_values('Coefficient', ascending=False)
    
    top_15_coef = coefficients.head(15)
    
    ax5.barh(range(len(top_15_coef)), top_15_coef['Coefficient'], color='coral')
    ax5.set_yticks(range(len(top_15_coef)))
    ax5.set_yticklabels(top_15_coef['Feature'], fontsize=8)
    ax5.set_xlabel('Coefficient Magnitude')
    ax5.set_title('Top 15 Features by Coefficient', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()  

# 6. Performance Metrics
ax6 = plt.subplot(2,3,6)
best_model_results = results_df[
    (results_df['Model'] == best_model_name) &
    (results_df['Augmentation'] == best_aug)
].iloc[0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
values = [best_model_results[m] for m in metrics]
bars = ax6.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax6.set_ylim([0, 1])
ax6.set_ylabel('Score')
ax6.set_title(f'Performance Metrics\n({best_model_name} + {best_aug})\n',
             fontsize=12, fontweight='bold')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax6.axhline(y=0.8, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('biomarker_classification_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'biomarker_classification_results.png'")


# CROSS-VALIDATION

print("\nCross-Validation...")
print("-"*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_best, y_best = augmentation_strategies[best_aug]
cv_scores = cross_val_score(best_model, X_best, y_best, cv=skf, scoring='roc_auc')

print(f"\n5-Fold Cross-Validation Results:")
print(f"ROC-AUC Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# SAVE RESULTS

results_df.to_csv('model_comparision_results.csv', index=False)
print("\nResults saved to 'model_comparision_results.csv'")

print("\n" + "="*80)
print("Biomarker Classification Completed.")
print("="*80)





