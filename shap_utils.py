"""
SHAP Explainability
-------------------
Creates SHAP plots for XGBoost.
"""

import shap
import joblib
import matplotlib.pyplot as plt

from config import MODEL_PATH


model = joblib.load(MODEL_PATH)


def create_explainer():
    """
    Create TreeExplainer.
    """

    explainer = shap.TreeExplainer(model)

    return explainer


def save_summary_plot(
        sample_data,
        save_path
):
    """
    Save SHAP Summary Plot.
    """

    explainer = create_explainer()

    shap_values = explainer.shap_values(
        sample_data
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        sample_data,
        show=False
    )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def save_waterfall_plot(
        sample_data,
        save_path
):
    """
    Save SHAP Waterfall Plot.
    """

    explainer = create_explainer()

    explanation = explainer(
        sample_data
    )

    plt.figure()

    shap.plots.waterfall(
        explanation[0],
        show=False
    )

    plt.savefig(save_path)

    plt.close()