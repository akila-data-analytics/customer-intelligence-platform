
# Customer Intelligence & Predictive Retention Platform - Visualizations

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.txt"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load project data
df = pd.read_csv(DATA_DIR / "telco_cleaned.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

model_results = pd.read_csv(DATA_DIR / "model_comparison.csv")
segments_df = pd.read_csv(DATA_DIR / "customer_segments.csv")

model = joblib.load(MODEL_PATH)

threshold = float(
    THRESHOLD_PATH.read_text(encoding="utf-8").strip()
)

# Ensure test features match the trained model
model_columns = list(model.feature_names_in_)

missing_columns = [
    column for column in model_columns
    if column not in X_test.columns
]

if missing_columns:
    raise ValueError(
        f"Missing test features required by model: {missing_columns}"
    )

X_test = X_test[model_columns].copy()
y_test = pd.Series(y_test).reset_index(drop=True)

sns.set_theme(style="whitegrid")

print("Selected model:", type(model).__name__)
print("Prediction threshold:", threshold)
print("Test data shape:", X_test.shape)


# 3. Helper function to save charts
def save_chart(filename):
    output_path = REPORTS_DIR / filename

    try:
        plt.tight_layout()
        plt.savefig(
            str(output_path),
            dpi=200,
            bbox_inches="tight",
            format="png"
        )
        print(f"Saved: {filename}")

    finally:
        plt.close("all")


# 4. Churned vs retained customers
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Churned vs Retained Customers")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

save_chart("churn_count.png")


# 5. Churn rate by contract type
contract_rate = (
    df.groupby("Contract", observed=False)["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
)

plt.figure(figsize=(8, 5))

contract_rate.plot(kind="bar")

plt.title("Churn Rate by Contract")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

save_chart("contract_churn.png")


# 6. Churn rate by internet plan
plan_rate = (
    df.groupby("InternetService", observed=False)["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
)

plt.figure(figsize=(8, 5))

plan_rate.plot(kind="bar")

plt.title("Churn Rate by Internet Plan")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

save_chart("plan_churn.png")


# 7. Monthly charges by churn status
plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")

save_chart("monthly_charges_by_churn.png")


# 8. Customer tenure by churn status
plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="Churn", y="tenure")

plt.title("Customer Tenure by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Tenure (Months)")

save_chart("tenure_by_churn.png")


# 9. Numerical feature correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Numerical Feature Correlation")

save_chart("correlation_heatmap.png")


# 10. Model performance comparison
metrics = [
    column for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]
    if column in model_results.columns
]

if metrics and "Model" in model_results.columns:
    plot_data = model_results.set_index("Model")[metrics]

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)

    save_chart("model_comparison.png")
else:
    print("Skipped model comparison: expected columns not found.")


# 11. Generate predictions using the saved threshold
probabilities_all = model.predict_proba(X_test)

classes = list(model.classes_)

if 1 not in classes:
    raise ValueError("Churn class 1 was not found in the model.")

churn_index = classes.index(1)

probabilities = probabilities_all[:, churn_index]

predictions = (probabilities >= threshold).astype(int)


# 12. Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stay", "Churn"]
)

disp.plot(cmap="Blues")

plt.title(
    f"{type(model).__name__} - Confusion Matrix "
    f"(Threshold = {threshold:.3f})"
)

save_chart("confusion_matrix.png")


# 13. ROC curve
fpr, tpr, _ = roc_curve(y_test, probabilities)

roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"{type(model).__name__} (AUC = {roc_auc_value:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random baseline"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

save_chart("roc_curve.png")


# 14. Precision-Recall curve
precision, recall, _ = precision_recall_curve(
    y_test,
    probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

save_chart("precision_recall_curve.png")


# 15. Feature importance for the selected model
if hasattr(model, "coef_"):
    importance_values = abs(model.coef_[0])
    importance_label = "Absolute Coefficient"

elif hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_
    importance_label = "Feature Importance"

else:
    importance_values = None
    print("Skipped feature importance: unsupported model type.")

if importance_values is not None:
    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importance_values
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        REPORTS_DIR / "feature_importance.csv",
        index=False
    )

    top_features = importance_df.head(15).sort_values(
        "Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(importance_label)
    plt.ylabel("Feature")
    plt.title("Top 15 Features Associated with Churn")

    save_chart("feature_importance.png")


# 16. Customer segment visualization
segment_column = next(
    (
        column for column in ["CustomerSegment", "Segment"]
        if column in segments_df.columns
    ),
    None
)

if (
    {"tenure", "MonthlyCharges"}.issubset(segments_df.columns)
    and segment_column
):
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=segments_df,
        x="tenure",
        y="MonthlyCharges",
        hue=segment_column,
        alpha=0.6
    )

    plt.title("Customer Segments")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Monthly Charges")

    save_chart("customer_segments.png")

else:
    print(
        "Skipped customer segment chart: expected columns not found."
    )


# 17. Churn risk categories
risk_data = X_test[["MonthlyCharges"]].copy()

risk_data["Churn_Probability"] = probabilities

risk_data["RiskCategory"] = pd.cut(
    probabilities,
    bins=[-0.001, 0.30, 0.60, 1.001],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Risk category counts
plt.figure(figsize=(8, 5))

sns.countplot(
    data=risk_data,
    x="RiskCategory",
    order=["Low Risk", "Medium Risk", "High Risk"]
)

plt.title("Customers by Churn Risk Category")
plt.xlabel("Risk Category")
plt.ylabel("Number of Test Customers")

save_chart("risk_category_counts.png")


# 18. Monthly charges associated with each risk category
revenue_exposure = (
    risk_data.groupby(
        "RiskCategory",
        observed=False
    )["MonthlyCharges"]
    .sum()
    .reindex(
        ["Low Risk", "Medium Risk", "High Risk"],
        fill_value=0
    )
)

plt.figure(figsize=(8, 5))

ax = revenue_exposure.plot(kind="bar")

ax.set_title("Monthly Charges by Churn Risk Category")
ax.set_xlabel("Risk Category")
ax.set_ylabel("Sum of Monthly Charges")
plt.xticks(rotation=0)

save_chart("revenue_exposure_by_risk.png")


# 19. Churn probability distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    risk_data["Churn_Probability"],
    bins=20,
    kde=True
)

plt.title("Churn Probability Distribution")
plt.xlabel("Predicted Churn Probability")
plt.ylabel("Number of Test Customers")

save_chart("churn_probability_distribution.png")


# 20. Completion summary
print("\nAll available visualization charts have been saved.")
print("Reports folder:", REPORTS_DIR)

print(
    "\nBusiness note: Monthly charges grouped by risk are an exposure "
    "indicator only. They are not measured revenue loss or guaranteed "
    "recoverable revenue."
)
