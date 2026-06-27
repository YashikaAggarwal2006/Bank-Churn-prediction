# Bank-Churn-prediction



A Machine Learning-based web application that predicts whether a bank customer is likely to leave the bank using customer information. The project compares multiple machine learning models and deploys the best-performing model (XGBoost) through a Streamlit dashboard with SHAP explainability.

---

## 📌 Features

- 📊 Data Preprocessing
- ⚖️ Handles Class Imbalance using SMOTE
- 🤖 Multiple Machine Learning Models
  - Logistic Regression
  - Random Forest
  - XGBoost (Best Model)
- 📈 ROC Curve & AUC Score Comparison
- 📉 Feature Importance Visualization
- 🔍 SHAP Explainability
- 🎯 Live Customer Churn Prediction
- 📥 Download Prediction Result as CSV
- 🌙 Professional Dark Mode Streamlit Dashboard

---

## 📂 Project Structure

```
Bank-Churn-Prediction/
│
├── app.py
├── train.py
├── predict.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Bank Customer Churn Prediction.csv
│
├── models/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
│
└── plots/
    ├── roc_curve.png
    ├── feature_importance.png
    ├── feature_importance.csv
    ├── model_comparison.csv
    ├── shap_summary.png
    └── waterfall.png
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SMOTE (Imbalanced-Learn)
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- SHAP
- Joblib

---

## 📊 Dataset

The dataset contains **10,000+ customer records** with the following features:

| Feature | Description |
|---------|-------------|
| customer_id | Unique Customer ID |
| credit_score | Customer Credit Score |
| country | Customer Country |
| gender | Gender |
| age | Age |
| tenure | Years with Bank |
| balance | Account Balance |
| products_number | Number of Bank Products |
| credit_card | Has Credit Card (0/1) |
| active_member | Active Member (0/1) |
| estimated_salary | Estimated Salary |
| churn | Target Variable |

---

## 🤖 Machine Learning Models

The project compares three classification algorithms.

| Model | Purpose |
|--------|----------|
| Logistic Regression | Baseline Model |
| Random Forest | Ensemble Learning |
| XGBoost | Final Deployed Model |

The best-performing model is selected based on the **ROC-AUC Score**.

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---


---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Bank-Churn-Prediction.git
```

Move inside the project

```bash
cd Bank-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

This generates:

- Trained XGBoost Model
- Feature Importance Plot
- ROC Curve
- SHAP Plots
- Model Comparison CSV

---

## 💻 Run the Dashboard

```bash
streamlit run app.py
```

---

## 📊 Dashboard

The Streamlit dashboard provides:

- 🏠 Home Dashboard
- 📊 Dataset Overview
- 🤖 Live Customer Prediction
- 📈 Model Comparison
- 📉 ROC Curve
- ⭐ Feature Importance
- 🔍 SHAP Explainability

---

## 📥 Prediction Output

The dashboard predicts whether the customer will:

- ✅ Stay
- ❌ Churn

It also displays:

- Churn Probability
- Probability Gauge
- Download Prediction as CSV

---

## 📸 Results

The project automatically saves:

- ROC Curve
- Feature Importance Plot
- SHAP Summary Plot
- SHAP Waterfall Plot
- Model Comparison Report

inside the `plots` folder.

---

## 🎯 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- LightGBM & CatBoost Comparison
- Docker Deployment
- Cloud Deployment
- User Authentication
- Batch Prediction

---

## 👩‍💻 Author

**Yashika Aggarwal**

B.Tech Computer Science Engineering

---

