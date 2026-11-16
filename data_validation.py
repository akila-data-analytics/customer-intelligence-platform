
# Data Validation - Customer Intelligence Project

import pandas as pd
from pathlib import Path

# Dataset file path
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"

# Check whether the dataset exists
if not file_path.exists():
    print("Dataset file not found!")
else:
    # Load the dataset
    data = pd.read_csv(file_path)

    print("=" * 50)
    print("CUSTOMER DATA VALIDATION REPORT")
    print("=" * 50)

    # Dataset size
    print("\n1. Dataset Shape")
    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])

    # Column names
    print("\n2. Column Names")
    print(data.columns.tolist())

    # Data types
    print("\n3. Data Types")
    print(data.dtypes)

    # Missing values
    print("\n4. Missing Values")
    print(data.isnull().sum())

    # Duplicate records
    print("\n5. Duplicate Records")
    print(data.duplicated().sum())

    # Unique customer IDs
    print("\n6. Unique Customer IDs")
    print(data["customerID"].nunique())

    # Churn distribution
    print("\n7. Churn Distribution")
    print(data["Churn"].value_counts())

    # Churn percentage
    print("\n8. Churn Percentage")
    print((data["Churn"].value_counts(normalize=True) * 100).round(2))

    print("\nData validation completed!")
