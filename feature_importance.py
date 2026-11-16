
# Customer Intelligence & Predictive Retention Platform - Feature Importance

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"
OUTPUT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load trained model and training features
model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(DATA_PATH)

# 3. Get feature importance based on the selected model
if hasattr(model, "coef_"):
    # Logistic Regression: absolute coefficient indicates strength
    importance_values = abs(model.coef_[0])
    importance_label = "Absolute Coefficient"

elif hasattr(model, "feature_importances_"):
    # Tree-based models: built-in feature importance
    importance_values = model.feature_importances_
    importance_label = "Feature Importance"

else:
    raise ValueError(
        "The selected model does not support this feature importance method."
    )

# 4. Check feature alignment
model_features = list(model.feature_names_in_)

if list(X_train.columns) != model_features:
    X_train = X_train[model_features]

if len(importance_values) != len(X_train.columns):
    raise ValueError(
        "Feature count does not match the model's importance values."
    )

# 5. Create feature importance table
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# 6. Display top 10 features
print("\nTOP 10 FEATURES ASSOCIATED WITH CHURN")
print(importance_df.head(10).to_string(index=False))

# 7. Save feature importance table
csv_path = OUTPUT_DIR / "feature_importance.csv"
importance_df.to_csv(csv_path, index=False)

# 8. Plot top 10 features
top_features = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(importance_label)
plt.ylabel("Customer Features")
plt.title("Top 10 Features Associated with Customer Churn")
plt.tight_layout()

# 9. Save chart
chart_path = OUTPUT_DIR / "feature_importance.png"

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 10. Completion message
print("\nFeature importance analysis completed successfully!")
print(f"CSV saved at: {csv_path}")
print(f"Chart saved at: {chart_path}")
print(
    "\nNote: For Logistic Regression, absolute coefficients show "
    "the strength of association, not whether a feature increases "
    "or decreases churn. Features are not standardized in this model, "
    "so coefficient magnitudes can be affected by feature scales."
)
