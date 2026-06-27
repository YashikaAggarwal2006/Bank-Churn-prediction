"""
=========================================================
Bank Customer Churn Prediction

Prediction Module

This file:

1. Loads saved model
2. Loads scaler
3. Loads encoders
4. Converts user input into model input
5. Predicts churn probability

=========================================================
"""

import joblib
import pandas as pd

from config import (
    MODEL_FILE,
    SCALER_FILE,
    ENCODER_FILE,
    FEATURE_FILE
)

# =========================================================
# Load Saved Files
# =========================================================

model = joblib.load(MODEL_FILE)

scaler = joblib.load(SCALER_FILE)

encoders = joblib.load(ENCODER_FILE)

feature_names = joblib.load(FEATURE_FILE)


# =========================================================
# Encode User Input
# =========================================================

def preprocess_input(customer_data):
    """
    Convert dictionary input into model-ready dataframe.
    """

    df = pd.DataFrame([customer_data])

    # Encode categorical columns
    for column in encoders:

        try:

            df[column] = encoders[column].transform(df[column])

        except:

            # If category not seen during training
            df[column] = 0

    # Arrange columns
    df = df[feature_names]

    return df


# =========================================================
# Predict
# =========================================================

def predict(customer_data):
    """
    Predict customer churn.

    Parameters
    ----------
    customer_data : dict

    Returns
    -------
    prediction
    probability
    """

    df = preprocess_input(customer_data)

    # XGBoost uses original features
    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability


# =========================================================
# Test Prediction
# =========================================================

if __name__ == "__main__":

    sample_customer = {

        "credit_score": 650,

        "country": "France",

        "gender": "Female",

        "age": 35,

        "tenure": 5,

        "balance": 50000,

        "products_number": 2,

        "credit_card": 1,

        "active_member": 1,

        "estimated_salary": 60000

    }

    prediction, probability = predict(sample_customer)

    print("=" * 50)

    if prediction == 1:

        print("Prediction : Customer Will Churn")

    else:

        print("Prediction : Customer Will Stay")

    print(f"Probability : {probability:.2%}")

    print("=" * 50)