# 🧠 MedPredict AI – Medical Diagnosis Prediction System

An AI-powered medical diagnosis prediction system that analyzes patient health data and predicts possible diseases using machine learning models.


## 🚀 Project Overview

This project uses machine learning algorithms to predict a patient's primary diagnosis based on medical attributes such as symptoms, vitals, and medical history.

It includes:

* Data preprocessing & analysis
* Multiple ML models
* Model comparison & evaluation
* Interactive web interface

---

## 📂 Project Structure

```
MedPredict-AI/
│
├── dataset/
│   └── medical_diagnosis_dataset_1000.xlsx
│
├── backend/
│   └── medical_diagnosis.py
│
├── frontend/
│   └── index.html
│
├── outputs/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
└── README.md
```

---

## 🧾 Dataset Details

* File: `medical_diagnosis_dataset_1000.xlsx`
* Records: ~1000 patients
* Features:

  * Age, BMI, Blood Pressure
  * Symptoms (Fever, Cough, Fatigue, etc.)
  * Medical indicators (HbA1c, Cholesterol)
* Target:

  * `Primary_Diagnosis` (Multi-class classification)

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* HTML, CSS (Frontend UI)

---

## 🧠 Machine Learning Models

The system trains and compares:

* Logistic Regression
* Random Forest
* Gradient Boosting

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

From your code: 
👉 Best model is selected based on **F1-score**, not just accuracy (good decision).

---

## 🔍 Features

* Data Cleaning & Preprocessing
* Label Encoding for categorical data
* Feature Scaling using StandardScaler
* Exploratory Data Analysis (EDA)
* Model Training & Comparison
* Confusion Matrix Visualization
* Feature Importance Analysis
* Interactive UI for predictions

---

## 📊 Output Visualizations

The project generates:

* Class Distribution Chart
* Age Distribution
* Correlation Heatmap
* BMI vs Diagnosis
* Model Comparison Graph
* Confusion Matrix
* Feature Importance

---

## 💻 How to Run the Project

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

---

### 2. Run the Python Model

```bash
python medical_diagnosis.py
```

---

### 3. Open Frontend

Open the HTML file in browser:

👉 

Just double-click it or use Live Server.

---

## 📈 Workflow

1. Load dataset
2. Clean and preprocess data
3. Encode categorical variables
4. Scale features
5. Train multiple ML models
6. Evaluate performance
7. Select best model
8. Visualize results
9. Predict diagnosis

---

## ⚠️ Limitations (Be honest, this matters)

* Dataset is synthetic (not real clinical data)
* No real-time API integration
* Model not validated for real medical use
* No deep learning / advanced models
* No deployment (yet)

---

## 🔥 Improvements You SHOULD Add (if you want this to stand out)

* Convert to Flask / FastAPI backend
* Connect frontend with real model API
* Add real-time prediction
* Use Deep Learning (CNN / ANN)
* Deploy using Render / Vercel + backend
* Use real medical dataset (if available)

---

## 📌 Future Scope

* Mobile app for doctors/patients
* Integration with hospital systems
* Real-time health monitoring
* Explainable AI (XAI) for predictions

