
# Model Training - Customer Intelligence Project

import pandas as pd
import joblib

from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve
)

# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load prepared datasets
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 3. Create training and validation sets
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# 4. Define baseline and four classifiers
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
    max_iter=10000,
    class_weight="balanced",
    random_state=42
),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = []
trained_models = {}
validation_thresholds = {}

# 5. Train models and select thresholds using validation data
for name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)

    model.fit(X_fit, y_fit)

    val_probabilities = model.predict_proba(X_val)[:, 1]

    if name == "Baseline":
        best_threshold = 0.50
    else:
        precision_values, recall_values, thresholds = (
            precision_recall_curve(y_val, val_probabilities)
        )

        # Calculate F1 for each available threshold
        f1_values = (
            2 * precision_values[:-1] * recall_values[:-1]
            / (
                precision_values[:-1] + recall_values[:-1] + 1e-10
            )
        )

        best_index = f1_values.argmax()
        best_threshold = float(thresholds[best_index])

    validation_predictions = (
        val_probabilities >= best_threshold
    ).astype(int)

    validation_f1 = f1_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    trained_models[name] = model
    validation_thresholds[name] = best_threshold

    print(f"Validation F1: {validation_f1:.4f}")
    print(f"Selected Threshold: {best_threshold:.4f}")

# 6. Select best classifier using validation F1
selection_results = []

for name, model in trained_models.items():

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (
        probabilities >= validation_thresholds[name]
    ).astype(int)

    selection_results.append({
        "Model": name,
        "Validation_F1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

selection_df = pd.DataFrame(selection_results)

best_model_name = selection_df.loc[
    selection_df["Validation_F1"].idxmax(), "Model"
]

best_threshold = validation_thresholds[best_model_name]

print("\nSelected Model:", best_model_name)
print(f"Selected Threshold: {best_threshold:.4f}")

# 7. Refit selected model on all training data
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# 8. Evaluate models on the untouched test dataset
for name, model in models.items():

    # Fit each comparison model on all training data
    if name != best_model_name:
        model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    threshold = validation_thresholds[name]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(y_test, probabilities)
    }

    results.append(metrics)

    print("\n" + "-" * 55)
    print("Test Results:", name)

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

# 9. Save model comparison results
results_df = pd.DataFrame(results)

# Sort by test F1 for easy reporting only.
# Model selection itself was based on validation F1.
results_df = results_df.sort_values(
    "F1_Score", ascending=False
)

results_path = DATA_DIR / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

# 10. Save the selected model and its threshold
model_path = MODEL_DIR / "best_churn_model.joblib"

joblib.dump(best_model, model_path)

threshold_path = MODEL_DIR / "best_threshold.txt"
threshold_path.write_text(
    f"{best_threshold:.8f}",
    encoding="utf-8"
)

print("\n" + "=" * 55)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("Selected Model:", best_model_name)
print(f"Selected Threshold: {best_threshold:.4f}")
print("Saved Model:", model_path)
print("Saved Threshold:", threshold_path)
print("Saved Comparison:", results_path)
print("\nModel comparison table:")
print(results_df.round(4).to_string(index=False))
