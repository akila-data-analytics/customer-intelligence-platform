
# ML Data Preparation

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "telco_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load feature-engineered dataset
df = pd.read_csv(INPUT_PATH)

print("Dataset Loaded:", df.shape)

# 3. Remove customer ID because it is not a useful prediction feature
df = df.drop(columns=["customerID"])

# 4. Convert target column into numeric values
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# 5. Separate input features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 6. Convert categorical columns into numeric columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 7. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Save prepared datasets
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_frame().to_csv(
    OUTPUT_DIR / "y_train.csv", index=False
)
y_test.to_frame().to_csv(
    OUTPUT_DIR / "y_test.csv", index=False
)

# 9. Display results
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

print("\nTraining Churn Distribution:")
print(y_train.value_counts())

print("\nTesting Churn Distribution:")
print(y_test.value_counts())

print("\nMissing Values in Training Data:", X_train.isnull().sum().sum())
print("Missing Values in Testing Data:", X_test.isnull().sum().sum())

print("\nML DATA PREPARATION COMPLETED SUCCESSFULLY!")
print("Prepared files saved in:", OUTPUT_DIR)
