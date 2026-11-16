
# Customer Intelligence & Predictive Retention Platform

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.txt"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load test data
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

# 3. Load saved model and selected threshold
model = load(MODEL_PATH)
threshold = float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())

print("Selected Model:", type(model).__name__)
print(f"Selected Threshold: {threshold:.4f}")

# 4. Predict churn probabilities and labels
y_probability = model.predict_proba(X_test)[:, 1]
y_pred = (y_probability >= threshold).astype(int)

# 5. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)
pr_auc = average_precision_score(y_test, y_probability)

# 6. Display evaluation results
print("\nCUSTOMER CHURN MODEL EVALUATION")
print("-" * 40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC-AUC", "PR-AUC"
    ],
    "Score": [
        accuracy, precision, recall,
        f1, roc_auc, pr_auc
    ]
})

metrics.to_csv(
    REPORT_DIR / "model_evaluation_metrics.csv",
    index=False
)

# 8. Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"],
    cmap="Blues"
)

plt.title(f"Customer Churn - Confusion Matrix (Threshold={threshold:.4f})")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# 9. Save ROC curve
RocCurveDisplay.from_predictions(y_test, y_probability)

plt.title("Customer Churn - ROC Curve")
plt.tight_layout()
plt.savefig(REPORT_DIR / "roc_curve.png", dpi=300)
plt.close()

# 10. Save classification report
report_text = classification_report(
    y_test, y_pred, zero_division=0
)

with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("Customer Churn Model Evaluation\n")
    file.write(f"Model: {type(model).__name__}\n")
    file.write(f"Threshold: {threshold:.4f}\n\n")
    file.write(report_text)

print("\nEvaluation completed successfully!")
print("Results saved in:", REPORT_DIR)
