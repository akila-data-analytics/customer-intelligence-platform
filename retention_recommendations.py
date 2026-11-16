
# Customer Intelligence & Predictive Retention Platform - Retention Recommendations

from pathlib import Path

import pandas as pd
import joblib

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.txt"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
TARGET_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"
OUTPUT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load saved model and threshold
model = joblib.load(MODEL_PATH)
threshold = float(
    THRESHOLD_PATH.read_text(encoding="utf-8").strip()
)

X_test = pd.read_csv(TEST_DATA_PATH)
y_test = pd.read_csv(TARGET_PATH).iloc[:, 0]

# 3. Predict churn probabilities and labels
churn_probabilities = model.predict_proba(X_test)[:, 1]
churn_predictions = (churn_probabilities >= threshold).astype(int)

# 4. Assign risk level
# These are business reporting categories, separate from the model threshold.
def get_risk_level(probability):
    if probability >= 0.60:
        return "High"
    elif probability >= 0.30:
        return "Medium"
    else:
        return "Low"


def get_recommendation(risk):
    if risk == "High":
        return "Prioritize customer contact and discuss suitable retention options"
    elif risk == "Medium":
        return "Monitor satisfaction and investigate possible service issues"
    else:
        return "Continue regular customer engagement"


# 5. Create customer risk report
report = pd.DataFrame({
    # These are test-row numbers, not original customer IDs.
    "Test_Record_ID": range(1, len(X_test) + 1),
    "Actual_Churn": y_test.values,
    "Predicted_Churn": churn_predictions,
    "Churn_Probability": churn_probabilities
})

report["Risk_Level"] = report["Churn_Probability"].apply(get_risk_level)
report["Recommendation"] = report["Risk_Level"].apply(get_recommendation)

# 6. Display risk summary
print("\nCUSTOMER RETENTION ANALYSIS")
print("-" * 40)
print("Selected model:", type(model).__name__)
print(f"Prediction threshold: {threshold:.4f}")

print("\nRisk Level Summary:")
print(report["Risk_Level"].value_counts().reindex(
    ["High", "Medium", "Low"], fill_value=0
))

print("\nTop 10 High-Risk Test Records:")
high_risk = report[
    report["Risk_Level"] == "High"
].sort_values(
    by="Churn_Probability", ascending=False
)

if high_risk.empty:
    print("No high-risk records found.")
else:
    print(
        high_risk.head(10)[
            ["Test_Record_ID", "Churn_Probability",
             "Risk_Level", "Recommendation"]
        ].to_string(index=False)
    )

# 7. Save report
output_path = OUTPUT_DIR / "retention_recommendations.csv"
report.to_csv(output_path, index=False)

print("\nRetention analysis completed successfully!")
print("Report saved to:", output_path)
