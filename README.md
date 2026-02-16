# ❤️ Heart Attack Prediction using Machine Learning & Explainable AI

## 📌 Project Overview

This project develops a **Heart Attack Prediction System** using machine learning algorithms on structured clinical data.
The aim is to assist doctors and healthcare providers in identifying patients at **high risk of heart attack** at an early stage.

The system not only predicts risk but also explains *why* the prediction was made using Explainable AI (SHAP), which is very important in medical applications.

---

## 🧠 Technologies & Libraries Used

### Data Processing

* **Pandas** – For loading and managing dataset
* **NumPy** – For numerical and array operations

### Visualization

* **Matplotlib** – Graph plotting
* **Seaborn** – Statistical data visualization

### Machine Learning Models

* **Scikit-Learn**

  * Train/Test splitting
  * Feature scaling
  * Evaluation metrics
  * Logistic Regression
  * Random Forest Classifier

* **XGBoost**

  * High-performance gradient boosting model

### Explainable AI

* **SHAP (SHapley Additive Explanations)**

  * Feature importance
  * Model interpretability
  * Understanding predictions

---

## 📦 Imported Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import shap
shap.initjs()
```

---

## ⚙️ Purpose of Each Library

| Library                | Role in Project                         |
| ---------------------- | --------------------------------------- |
| pandas                 | Reads and handles medical dataset       |
| numpy                  | Mathematical processing                 |
| matplotlib             | Plotting performance graphs             |
| seaborn                | Visual data analysis                    |
| train_test_split       | Divides dataset into training & testing |
| StandardScaler         | Normalizes patient features             |
| LogisticRegression     | Baseline prediction model               |
| RandomForestClassifier | Tree-based ensemble model               |
| XGBClassifier          | Advanced boosting model                 |
| sklearn.metrics        | Evaluates model performance             |
| SHAP                   | Explains AI decisions                   |

---

## 🔬 Machine Learning Workflow

1. Load heart attack dataset
2. Perform data cleaning and preprocessing
3. Split dataset into training and testing data
4. Apply feature scaling (Standardization)
5. Train models:

   * Logistic Regression
   * Random Forest
   * XGBoost
6. Evaluate models
7. Interpret predictions using SHAP Explainable AI

---

## 📊 Model Evaluation Methods

The following metrics are used to check model performance:

* **Accuracy Score** – Overall correct predictions
* **Confusion Matrix** – TP, TN, FP, FN
* **Classification Report**

  * Precision
  * Recall
  * F1 Score
* **ROC Curve**
* **AUC Score**

---

## 🤖 Explainable AI (SHAP)

In healthcare, predictions must be explainable.
We use SHAP to:

* Show which features contributed to heart attack risk
* Help doctors trust the AI system
* Detect important clinical factors
* Improve transparency of the model

`shap.initjs()` enables interactive SHAP visualizations in Jupyter Notebook.

---

## 🎯 Objective

To build a **clinical decision support tool** that:

* Predicts heart attack risk early
* Supports medical professionals
* Improves patient monitoring
* Enables preventive treatment

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

### 2️⃣ Run the Program

If
