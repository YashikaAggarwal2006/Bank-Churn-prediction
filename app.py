"""
=========================================================
Bank Customer Churn Prediction Dashboard

Features
--------
✔ Home
✔ Dataset Overview
✔ Live Prediction
✔ Model Comparison
✔ ROC Curve
✔ Feature Importance
✔ SHAP Explainability

=========================================================
"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from predict import predict

from config import (
    DATA_PATH,
    PLOT_DIR
)

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(

    page_title="Bank Customer Churn Prediction",

    page_icon="🏦",

    layout="wide"

)

# =====================================================
# Custom CSS
# =====================================================

st.markdown("""

<style>

.main{

background:#0E1117;

color:white;

}

section[data-testid="stSidebar"]{

background:#161B22;

}

.stButton>button{

background:#4CAF50;

color:white;

border-radius:10px;

height:45px;

font-size:18px;

width:100%;

}

div[data-testid="metric-container"]{

background:#1F2937;

padding:18px;

border-radius:12px;

border:1px solid #2E3B4E;

}

h1,h2,h3,h4{

color:white;

}

</style>

""",unsafe_allow_html=True)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(DATA_PATH)

# =====================================================
# Sidebar
# =====================================================

st.sidebar.image(

    "https://img.icons8.com/color/96/bank-building.png",

    width=80

)

st.sidebar.title("🏦 Bank Churn")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📊 Dataset",

        "🤖 Live Prediction",

        "📈 Model Comparison",

        "📉 ROC Curve",

        "⭐ Feature Importance",

        "🔍 SHAP Explainability"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Model : XGBoost")

st.sidebar.info("ROC-AUC Based Selection")

# =====================================================
# Home
# =====================================================

if page=="🏠 Home":

    st.title("🏦 Bank Customer Churn Prediction")

    st.markdown("""

### AI Powered Customer Churn Prediction

This dashboard predicts whether a customer is likely to leave the bank.

### Features

- 🤖 Live Prediction

- 📈 Compare ML Models

- 📉 ROC Curve

- ⭐ Feature Importance

- 🔍 SHAP Explainability

- 📥 Download Prediction

Final Model :

## XGBoost

""")

    col1,col2,col3,col4=st.columns(4)

    with col1:

        st.info(

            f"👥 Customers\n\n### {len(df)}"

        )

    with col2:

        stayed=(df["churn"]==0).sum()

        st.success(

            f"😊 Stayed\n\n### {stayed}"

        )

    with col3:

        churn=(df["churn"]==1).sum()

        st.error(

            f"🚪 Churned\n\n### {churn}"

        )

    with col4:

        rate=100*df["churn"].mean()

        st.warning(

            f"📊 Churn Rate\n\n### {rate:.2f}%"

        )

    st.markdown("---")

    fig=px.histogram(

        df,

        x="age",

        color="churn",

        barmode="overlay",

        title="Customer Age Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# =====================================================
# Dataset
# =====================================================

elif page=="📊 Dataset":

    st.title("📊 Dataset Overview")

    st.subheader("Dataset")

    st.dataframe(

        df,

        use_container_width=True

    )

    c1,c2=st.columns(2)

    with c1:

        st.subheader("Missing Values")

        st.dataframe(

            df.isnull().sum()

        )

    with c2:

        st.subheader("Data Types")

        st.dataframe(

            df.dtypes

        )

    st.subheader("Statistics")

    st.dataframe(

        df.describe(),

        use_container_width=True

    )

    fig=px.pie(

        df,

        names="churn",

        title="Customer Churn Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # =====================================================
# Live Prediction
# =====================================================

elif page == "🤖 Live Prediction":

    st.title("🤖 Live Customer Churn Prediction")

    st.write("Enter customer information below.")

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            300,
            900,
            650
        )

        country = st.selectbox(
            "Country",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        age = st.slider(
            "Age",
            18,
            95,
            35
        )

        tenure = st.slider(
            "Tenure",
            0,
            10,
            5
        )

    with col2:

        balance = st.number_input(
            "Balance",
            0.0,
            300000.0,
            50000.0
        )

        products = st.slider(
            "Products Number",
            1,
            4,
            2
        )

        credit_card = st.selectbox(
            "Has Credit Card",
            [1,0]
        )

        active_member = st.selectbox(
            "Active Member",
            [1,0]
        )

        salary = st.number_input(
            "Estimated Salary",
            0.0,
            250000.0,
            60000.0
        )

    st.markdown("---")

    if st.button("Predict Churn"):

        customer = {

            "credit_score": credit_score,

            "country": country,

            "gender": gender,

            "age": age,

            "tenure": tenure,

            "balance": balance,

            "products_number": products,

            "credit_card": credit_card,

            "active_member": active_member,

            "estimated_salary": salary

        }

        prediction, probability = predict(customer)

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("🚨 Customer is likely to Churn")

        else:

            st.success("✅ Customer is likely to Stay")

        st.markdown("### Churn Probability")

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability*100,

                title={"text":"Probability (%)"},

                gauge={

                    "axis":{"range":[0,100]},

                    "bar":{"color":"red"},

                    "steps":[

                        {"range":[0,40],"color":"green"},

                        {"range":[40,70],"color":"orange"},

                        {"range":[70,100],"color":"red"}

                    ]

                }

            )

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        prediction_df = pd.DataFrame({

            "Prediction":[

                "Churn" if prediction==1 else "Stay"

            ],

            "Probability (%)":[

                round(probability*100,2)

            ]

        })

        st.dataframe(

            prediction_df,

            use_container_width=True

        )

        csv = prediction_df.to_csv(index=False)

        st.download_button(

            label="⬇ Download Prediction",

            data=csv,

            file_name="prediction.csv",

            mime="text/csv"

        )

# =====================================================
# Model Comparison
# =====================================================

elif page == "📈 Model Comparison":

    st.title("📈 Model Comparison")

    comparison = pd.read_csv(

        os.path.join(

            PLOT_DIR,

            "model_comparison.csv"

        )

    )

    st.dataframe(

        comparison,

        use_container_width=True

    )

    best_model = comparison.iloc[0]["Model"]

    best_auc = comparison.iloc[0]["ROC AUC"]

    col1,col2 = st.columns(2)

    with col1:

        st.success(

            f"🏆 Best Model\n\n### {best_model}"

        )

    with col2:

        st.info(

            f"🎯 ROC-AUC\n\n### {best_auc:.3f}"

        )

    fig = px.bar(

        comparison,

        x="Model",

        y="ROC AUC",

        color="Model",

        text="ROC AUC",

        title="ROC-AUC Comparison"

    )

    fig.update_traces(

        texttemplate="%{text:.3f}"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    fig2 = px.bar(

        comparison,

        x="Model",

        y=[

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"

        ],

        barmode="group",

        title="Performance Comparison"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    # =====================================================
# ROC Curve
# =====================================================

elif page == "📉 ROC Curve":

    st.title("📉 ROC Curve")

    roc_path = os.path.join(
        PLOT_DIR,
        "roc_curve.png"
    )

    if os.path.exists(roc_path):

        st.image(
            roc_path,
            caption="ROC Curve Comparison",
            use_container_width=True
        )

    else:

        st.warning("Run train.py first.")

# =====================================================
# Feature Importance
# =====================================================

elif page == "⭐ Feature Importance":

    st.title("⭐ XGBoost Feature Importance")

    image_path = os.path.join(
        PLOT_DIR,
        "feature_importance.png"
    )

    csv_path = os.path.join(
        PLOT_DIR,
        "feature_importance.csv"
    )

    if os.path.exists(image_path):

        st.image(
            image_path,
            use_container_width=True
        )

    if os.path.exists(csv_path):

        st.subheader("Top Features")

        importance = pd.read_csv(csv_path)

        st.dataframe(
            importance,
            use_container_width=True
        )

# =====================================================
# SHAP Explainability
# =====================================================

elif page == "🔍 SHAP Explainability":

    st.title("🔍 SHAP Explainability")

    st.info("""

SHAP explains how every feature contributes
towards the final prediction.

Positive SHAP Value
➡ Pushes customer towards Churn

Negative SHAP Value
➡ Pushes customer towards Stay

Longer bars indicate higher impact.

""")

    summary = os.path.join(

        PLOT_DIR,

        "shap_summary.png"

    )

    waterfall = os.path.join(

        PLOT_DIR,

        "waterfall.png"

    )

    if os.path.exists(summary):

        st.subheader("SHAP Summary")

        st.image(

            summary,

            use_container_width=True

        )

    else:

        st.warning("Summary plot not found.")

    if os.path.exists(waterfall):

        st.subheader("Waterfall Plot")

        st.image(

            waterfall,

            use_container_width=True

        )

    else:

        st.warning("Waterfall plot not found.")


# Footer


st.markdown("---")

st.markdown("""

<center>

### 🏦 Bank Customer Churn Prediction

Developed using

<b>Python • Streamlit • Scikit-Learn • XGBoost • SHAP</b>



</center>

""",

unsafe_allow_html=True
)