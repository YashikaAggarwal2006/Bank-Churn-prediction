"""
=========================================================
Bank Customer Churn Prediction Project

Configuration File

This file stores all project paths and constants.

=========================================================
"""

import os


# Project Paths


# Root directory of project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Bank Customer Churn Prediction.csv"
)

# Models Folder
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# Plots Folder
PLOT_DIR = os.path.join(
    BASE_DIR,
    "plots"
)


# Saved Files


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgb_model.pkl"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "label_encoders.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)


# Create Folders Automatically


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# Target Column


TARGET_COLUMN = "churn"


# Columns to Remove


DROP_COLUMNS = [
    "customer_id"
]


# Random State


RANDOM_STATE = 42


# Train Test Split


TEST_SIZE = 0.20


# XGBoost Parameters


XGB_PARAMS = {

    "n_estimators": 300,

    "learning_rate": 0.05,

    "max_depth": 5,

    "subsample": 0.9,

    "colsample_bytree": 0.8,

    "eval_metric": "logloss",

    "random_state": RANDOM_STATE

}


# Random Forest Parameters


RF_PARAMS = {

    "n_estimators": 200,

    "max_depth": 10,

    "random_state": RANDOM_STATE

}


# Logistic Regression Parameters
#

LR_PARAMS = {

    "max_iter": 1000,

    "random_state": RANDOM_STATE

}