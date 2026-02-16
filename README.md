# HEART ATTACK
## 📌 Overview

This project builds a **machine learning based disease prediction / classification system** using structured medical data.
We train multiple models and evaluate their performance using standard metrics and visualization techniques.
We also apply **Explainable AI (XAI)** using SHAP to understand how the model makes decisions.

---

## 🧠 Technologies & Libraries Used

### Data Handling

* **Pandas** – Data loading and preprocessing
* **NumPy** – Numerical computations

### Visualization

* **Matplotlib** – Graph plotting
* **Seaborn** – Statistical visualizations

### Machine Learning

* **Scikit-Learn**

  * Train-test splitting
  * Data standardization
  * Evaluation metrics
  * Logistic Regression
  * Random Forest

* **XGBoost**

  * Advanced gradient boosting classifier

### Explainable AI

* **SHAP (SHapley Additive Explanations)**

  * Model interpretability
  * Feature importance
  * Decision understanding

---

## 📦 Imported Modules

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

## ⚙️ What Each Library Does

| Library                | Purpose                               |
| ---------------------- | ------------------------------------- |
| pandas                 | Reads and manages dataset             |
| numpy                  | Mathematical operations               |
| matplotlib             | Plots graphs                          |
| seaborn                | Better visual analytics               |
| train_test_split       | Splits data into training and testing |
| StandardScaler         | Normalizes features                   |
| LogisticRegression     | Baseline classification model         |
| RandomForestClassifier | Ensemble decision tree model          |
| XGBClassifier          | Boosting based high-performance model |
| metrics                | Evaluate model performance            |
| SHAP                   | Explain model predictions             |

---

## 🔬 Machine Learning Workflow

1. Load dataset
2. Clean & preprocess data
3. Split into training and testing sets
4. Normalize features
5. Train multiple models:

   * Logistic Regression
   * Random Forest
   * XGBoost
6. Evaluate models using:

   * Accuracy
   * Confusion Matrix
   * Classification Report
   * ROC Curve
   * AUC Score
7. Interpret predictions using SHAP

---

## 📊 Evaluation Metrics

* **Accuracy Score** – Overall correctness
* **Confusion Matrix** – Correct vs incorrect predictions
* **Classification Report**

  * Precision
  * Recall
  * F1-score
* **ROC Curve**
* **AUC Score**

---

## 🤖 Explainable AI (SHAP)

We use SHAP to:

* Identify important medical features
* Understand model decisions
* Increase doctor trust
* Improve transparency

`shap.initjs()` enables interactive visual explanations inside Jupyter Notebook.

---

## 🎯 Project Goal

To create a **doctor-assist AI system** that:

* Predicts disease risk
* Supports early diagnosis
* Provides interpretable results
* Helps hospitals and clinics

---

## ▶️ How to Run

1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

2. Run the notebook or Python file

```bash
python main.py
```

---

## 📌 Future Improvements

* Add deep learning models
* Add medical imaging analysis
* Deploy as web or mobile app
* Integrate with hospital PACS systems
* Real-time prediction API

---

## 👨‍💻 Author

Manthan – Medical AI Research Student

