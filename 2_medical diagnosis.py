"""
========================================================
MEDICAL DIAGNOSIS PREDICTION - COMPLETE PYTHON CODE
========================================================
Dataset: medical_diagnosis_dataset_1000.xlsx
Target: Primary_Diagnosis (10 classes)
Tools: pandas, numpy, matplotlib, seaborn, scikit-learn
========================================================
"""

# ─────────────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# ─────────────────────────────────────────────────────
# STEP 2: LOAD DATASET
# ─────────────────────────────────────────────────────
df = pd.read_excel('medical_diagnosis_dataset_1000.xlsx')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape       : {df.shape}")
print(f"Rows        : {df.shape[0]}")
print(f"Columns     : {df.shape[1]}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 Rows:\n{df.head()}")

# ─────────────────────────────────────────────────────
# STEP 3: DATA PREPROCESSING
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# 3a. Check Missing Values
print(f"\nMissing Values:\n{df.isnull().sum()}")
# Result: No missing values in this dataset

# 3b. Check & Remove Duplicates
print(f"\nDuplicate Rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

# 3c. Drop unnecessary columns
# Patient_ID is just an identifier, Recommended_Test would cause data leakage
df_clean = df.drop(columns=['Patient_ID', 'Recommended_Test'])
print(f"\nShape after dropping ID & Recommended_Test: {df_clean.shape}")

# 3d. Encode Categorical Columns (Label Encoding)
# Why? Machine learning models require numerical input.
categorical_cols = ['Gender', 'Smoking_Status', 'Family_History',
                    'Fever', 'Cough', 'Fatigue', 'Chest_Pain',
                    'Shortness_of_Breath', 'Headache', 'Nausea',
                    'Abdominal_Pain', 'Severity']

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    le_dict[col] = le
    print(f"  Encoded: {col} → {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode target column
le_target = LabelEncoder()
df_clean['Primary_Diagnosis'] = le_target.fit_transform(df_clean['Primary_Diagnosis'])
print(f"\nTarget Classes: {dict(enumerate(le_target.classes_))}")

# ─────────────────────────────────────────────────────
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Summary Statistics
print("\nSummary Statistics (Numeric Features):")
print(df[['Age', 'BMI', 'Systolic_BP_mmHg', 'Fasting_Blood_Sugar_mg_dL',
           'HbA1c_%', 'Cholesterol_mg_dL']].describe().round(2))

# Class Distribution
print("\nClass Distribution:")
print(df['Primary_Diagnosis'].value_counts())

# 4a. Class Distribution Bar Chart
plt.figure(figsize=(10, 5))
counts = df['Primary_Diagnosis'].value_counts()
sns.barplot(x=counts.index, y=counts.values, palette='Set2')
plt.title('Distribution of Primary Diagnosis', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()
print("Chart saved: class_distribution.png")

# 4b. Age Distribution
plt.figure(figsize=(8, 4))
plt.hist(df['Age'], bins=20, color='#028090', edgecolor='white')
plt.title('Age Distribution of Patients', fontsize=14, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('age_distribution.png', dpi=150)
plt.show()

# 4c. Correlation Heatmap
numeric_cols = ['Age', 'BMI', 'Systolic_BP_mmHg', 'Diastolic_BP_mmHg',
                'Heart_Rate_bpm', 'Temperature_C', 'Fasting_Blood_Sugar_mg_dL',
                'HbA1c_%', 'Cholesterol_mg_dL', 'Oxygen_Saturation_%', 'Hemoglobin_g_dL']
plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
plt.title('Correlation Heatmap of Numeric Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

# 4d. BMI by Diagnosis
plt.figure(figsize=(11, 5))
top_diagnoses = df['Primary_Diagnosis'].value_counts().index[:6]
data_bmi = [df[df['Primary_Diagnosis'] == d]['BMI'].values for d in top_diagnoses]
bp = plt.boxplot(data_bmi, labels=top_diagnoses, patch_artist=True)
colors = sns.color_palette("Set2", len(top_diagnoses))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.title('BMI Distribution by Diagnosis', fontsize=13, fontweight='bold')
plt.xlabel('Diagnosis')
plt.ylabel('BMI')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('bmi_by_diagnosis.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────
# STEP 5: FEATURE SELECTION & SCALING
# ─────────────────────────────────────────────────────
X = df_clean.drop(columns=['Primary_Diagnosis'])
y = df_clean['Primary_Diagnosis']

# Feature Scaling (StandardScaler normalizes data to mean=0, std=1)
# Why? Logistic Regression is sensitive to feature scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeature Matrix Shape: {X_scaled.shape}")
print(f"Target Vector Shape : {y.shape}")
print(f"Number of Classes   : {len(np.unique(y))}")

# ─────────────────────────────────────────────────────
# STEP 6: TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────
# 80% training, 20% testing, stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining Set Size : {X_train.shape[0]} samples")
print(f"Testing Set Size  : {X_test.shape[0]} samples")

# ─────────────────────────────────────────────────────
# STEP 7: MODEL TRAINING (3 Models)
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': report['weighted avg']['precision'],
        'recall':    report['weighted avg']['recall'],
        'f1':        report['weighted avg']['f1-score'],
        'model':     model,
        'y_pred':    y_pred
    }
    print(f"  Accuracy : {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall   : {results[name]['recall']:.4f}")
    print(f"  F1-Score : {results[name]['f1']:.4f}")

# ─────────────────────────────────────────────────────
# STEP 8: MODEL EVALUATION
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL EVALUATION SUMMARY")
print("=" * 60)

eval_df = pd.DataFrame({
    name: {
        'Accuracy':  f"{r['accuracy']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall':    f"{r['recall']:.4f}",
        'F1-Score':  f"{r['f1']:.4f}"
    }
    for name, r in results.items()
}).T
print(eval_df.to_string())

# Model Comparison Chart
fig, ax = plt.subplots(figsize=(9, 5))
model_names = list(results.keys())
metric_keys = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(model_names))
width = 0.2
colors = ['#028090', '#F96167', '#2C5F2D', '#6D2E46']
for i, metric in enumerate(metric_keys):
    vals = [results[m][metric] for m in model_names]
    ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=colors[i])
ax.set_title('Model Comparison - Performance Metrics', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────
# STEP 9: BEST MODEL SELECTION & CONFUSION MATRIX
# ─────────────────────────────────────────────────────
best_model_name = max(results, key=lambda x: results[x]['f1'])
best = results[best_model_name]
print(f"\n✅ Best Model: {best_model_name}")
print(f"   F1-Score : {best['f1']:.4f}")
print(f"   Accuracy : {best['accuracy']:.4f}")

# Detailed Classification Report
print(f"\nDetailed Classification Report ({best_model_name}):")
print(classification_report(y_test, best['y_pred'], target_names=le_target.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Feature Importance (Random Forest)
rf_model = results['Random Forest']['model']
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
plt.figure(figsize=(9, 5))
feat_imp.plot(kind='barh', color=sns.color_palette('Blues_r', 12))
plt.title('Top 12 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("PROJECT COMPLETE! All charts saved successfully.")
print("=" * 60)
