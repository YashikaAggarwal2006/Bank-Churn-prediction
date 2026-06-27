"""
Evaluation Module
-----------------
Generates confusion matrix and classification report.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from config import PLOT_DIR


def save_confusion_matrix(
        y_true,
        y_pred
):
    """
    Save confusion matrix plot.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/confusion_matrix.png"
    )

    plt.close()


def print_report(
        y_true,
        y_pred
):
    """
    Print classification report.
    """

    print(classification_report(
        y_true,
        y_pred
    ))