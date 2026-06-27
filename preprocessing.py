"""
Data Preprocessing Module
-------------------------
This file performs:

1. Read dataset
2. Handle missing values
3. Encode categorical columns
4. Split data
5. Scale numerical features
"""

import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import joblib

from config import (
    DATA_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURE_PATH
)


def load_dataset():
    """
    Load dataset from data folder.

    Returns
    -------
    dataframe
    """

    df = pd.read_csv(DATA_PATH)

    return df


def preprocess_data(df):
    """
    Perform preprocessing.

    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    # Remove customer id columns if available
    drop_columns = [
        "RowNumber",
        "CustomerId",
        "Surname"
    ]

    existing = [col for col in drop_columns if col in df.columns]

    if len(existing) > 0:
        df = df.drop(columns=existing)

    # Target column
    target = "Exited"

    X = df.drop(target, axis=1)
    y = df[target]

    encoders = {}

    # Encode categorical variables
    for column in X.select_dtypes(include="object").columns:

        encoder = LabelEncoder()

        X[column] = encoder.fit_transform(X[column])

        encoders[column] = encoder

    # Save encoders
    joblib.dump(encoders, ENCODER_PATH)

    # Save feature names
    joblib.dump(list(X.columns), FEATURE_PATH)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


if __name__ == "__main__":

    dataframe = load_dataset()

    preprocess_data(dataframe)

    print("Preprocessing Complete")