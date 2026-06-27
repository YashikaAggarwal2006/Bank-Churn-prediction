import pandas as pd
from src.preprocessing import load_dataset, preprocess_data
from src.shap_utils import save_summary_plot, save_waterfall_plot

df = load_dataset()

X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

save_summary_plot(
    X_test,
    "plots/shap_summary.png"
)

save_waterfall_plot(
    X_test.iloc[[0]],
    "plots/waterfall.png"
)

print("SHAP plots created successfully.")