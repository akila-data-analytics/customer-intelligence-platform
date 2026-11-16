
# Data Cleaning

import pandas as pd
from pathlib import Path

# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load original dataset
df = pd.read_csv(INPUT_PATH)

print("Original Dataset Shape:", df.shape)

# 3. Remove duplicate records
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

print("Duplicate Records Removed:", duplicate_count)

# 4. Clean column names
df.columns = df.columns.str.strip()

# 5. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("Missing TotalCharges Before Cleaning:",
      df["TotalCharges"].isnull().sum())

# 6. Handle missing TotalCharges
# New customers with zero tenure may have blank TotalCharges.
zero_tenure = df["tenure"] == 0

df.loc[zero_tenure, "TotalCharges"] = (
    df.loc[zero_tenure, "TotalCharges"].fillna(0)
)

# Handle any remaining missing TotalCharges using MonthlyCharges
remaining_missing = df["TotalCharges"].isnull()

df.loc[remaining_missing, "TotalCharges"] = (
    df.loc[remaining_missing, "MonthlyCharges"]
)

print("Missing Values After Cleaning:")
print(df.isnull().sum())

# 7. Check final dataset
print("\nCleaned Dataset Shape:", df.shape)

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# 8. Save cleaned dataset
output_path = OUTPUT_DIR / "telco_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")
print("Cleaned dataset saved to:", output_path)
