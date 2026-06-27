"""
=========================================================
Bank Customer Churn Prediction

Model Training Script

This script:

1. Loads dataset
2. Cleans data
3. Encodes categorical variables
4. Splits train/test
5. Scales features
6. Handles class imbalance using SMOTE

=========================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import shap
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report

)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from config import *


# Load Dataset


print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv(DATA_PATH)

print(df.head())

print("\nDataset Shape :", df.shape)


# Remove Unwanted Columns


for column in DROP_COLUMNS:

    if column in df.columns:

        df.drop(column, axis=1, inplace=True)

print("\nColumns After Cleaning")

print(df.columns)


# Missing Values


print("\nMissing Values")

print(df.isnull().sum())


# Separate Features & Target


X = df.drop(TARGET_COLUMN, axis=1)

y = df[TARGET_COLUMN]


# Encode Categorical Columns


encoders = {}

categorical_columns = X.select_dtypes(include="object").columns

print("\nCategorical Columns")

print(categorical_columns)

for column in categorical_columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(X[column])

    encoders[column] = encoder

joblib.dump(
    encoders,
    ENCODER_FILE
)

print("\nEncoding Completed")


# Save Feature Names


feature_names = list(X.columns)

joblib.dump(
    feature_names,
    FEATURE_FILE
)


# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples :", len(X_test))


# Feature Scaling


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    SCALER_FILE
)

print("\nScaling Completed")


# Handle Class Imbalance


print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=RANDOM_STATE
)

X_train_smote, y_train_smote = smote.fit_resample(

    X_train,

    y_train

)

X_train_scaled_smote, y_train_scaled_smote = smote.fit_resample(

    X_train_scaled,

    y_train

)

print("Before SMOTE")

print(y_train.value_counts())

print("\nAfter SMOTE")

print(pd.Series(y_train_smote).value_counts())


# ModeL


models = {

    "Logistic Regression":

        LogisticRegression(**LR_PARAMS),

    "Random Forest":

        RandomForestClassifier(**RF_PARAMS),

    "XGBoost":

        XGBClassifier(**XGB_PARAMS)

}

results = []

best_auc = 0

best_model = None

best_name = ""


# Train Models


print("\n" + "="*60)
print("Training Models...")
print("="*60)

plt.figure(figsize=(8,6))

for name, model in models.items():

    print(f"\n{name}")

    print("-"*50)

    # Logistic Regression uses scaled data
    if name == "Logistic Regression":

        model.fit(
            X_train_scaled_smote,
            y_train_scaled_smote
        )

        predictions = model.predict(
            X_test_scaled
        )

        probability = model.predict_proba(
            X_test_scaled
        )[:,1]

    # Tree models use original data
    else:

        model.fit(
            X_train_smote,
            y_train_smote
        )

        predictions = model.predict(
            X_test
        )

        probability = model.predict_proba(
            X_test
        )[:,1]

  
    # Metrics


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")

  
    # Classification Report


    print("\nClassification Report")

    print(

        classification_report(

            y_test,

            predictions

        )

    )


    # Confusion Matrix


    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(5,4))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(f"{name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            f"{name.lower().replace(' ','_')}_cm.png"

        )

    )

    plt.close()


    # ROC Curve


    fpr, tpr, _ = roc_curve(

        y_test,

        probability

    )

    plt.figure(1)

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"{name} (AUC={auc:.3f})"

    )


    # Store Results
    

    results.append({

        "Model":name,

        "Accuracy":round(accuracy,4),

        "Precision":round(precision,4),

        "Recall":round(recall,4),

        "F1 Score":round(f1,4),

        "ROC AUC":round(auc,4)

    })


    # Best Model


    if auc > best_auc:

        best_auc = auc

        best_model = model

        best_name = name

print("\n")

print("="*60)

print("Training Completed")

print("="*60)


# Save ROC Curve


plt.figure(1)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--",

    color="black"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOT_DIR,

        "roc_curve.png"

    )

)

plt.close()

print("ROC Curve Saved")


# Model Comparison Table


comparison = pd.DataFrame(results)

comparison = comparison.sort_values(

    by="ROC AUC",

    ascending=False

)

print("\nModel Comparison\n")

print(comparison)

comparison.to_csv(

    os.path.join(

        PLOT_DIR,

        "model_comparison.csv"

    ),

    index=False

)

print("\nComparison CSV Saved")


# Save Best Model


joblib.dump(

    best_model,

    MODEL_FILE

)

print(f"\nBest Model Saved : {best_name}")

print(f"ROC AUC : {best_auc:.4f}")

# Feature Importance


print("\nGenerating Feature Importance...")

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": best_model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    # Save CSV
    importance.to_csv(

        os.path.join(

            PLOT_DIR,

            "feature_importance.csv"

        ),

        index=False

    )

    # Plot
    plt.figure(figsize=(10,6))

    sns.barplot(

        data=importance.head(10),

        x="Importance",

        y="Feature",

        palette="viridis"

    )

    plt.title("Top 10 Important Features")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "feature_importance.png"

        )

    )

    plt.close()

    print("Feature Importance Saved")


# SHAP Explainability


print("\nGenerating SHAP Plots...")

try:

    explainer = shap.TreeExplainer(best_model)

    shap_values = explainer.shap_values(X_test)

    # Summary Plot
    plt.figure()

    shap.summary_plot(

        shap_values,

        X_test,

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "shap_summary.png"

        )

    )

    plt.close()

    # Waterfall Plot
    explanation = explainer(X_test)

    plt.figure()

    shap.plots.waterfall(

        explanation[0],

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "waterfall.png"

        )

    )

    plt.close()

    print("SHAP Plots Saved")

except Exception as e:

    print("\nSHAP could not be generated.")

    print(e)

# =========================================================
# Final Output
# =========================================================

print("\n" + "="*60)

print("Training Completed Successfully!")

print("="*60)

print("\nFiles Generated:")

print("models/xgb_model.pkl")

print(" models/scaler.pkl")

print("models/label_encoders.pkl")

print(" models/feature_names.pkl")

print("plots/model_comparison.csv")

print("plots/roc_curve.png")

print(" plots/feature_importance.csv")

print("plots/feature_importance.png")

print(" plots/shap_summary.png")

print(" plots/waterfall.png")

print("\nBest Model :", best_name)

print("ROC-AUC :", round(best_auc,4))

print("\nReady for Streamlit Dashboard 🚀")