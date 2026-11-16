
# Feature Engineering

import pandas as pd
from pathlib import Path

# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "telco_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load cleaned dataset
df = pd.read_csv(INPUT_PATH)

print("Cleaned Dataset Loaded:", df.shape)

# 3. Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "New Customer",
        "1-2 Years",
        "2-4 Years",
        "4-6 Years"
    ]
)

# 4. Create average monthly spend
df["AverageMonthlySpend"] = 0.0

has_tenure = df["tenure"] > 0

df.loc[has_tenure, "AverageMonthlySpend"] = (
    df.loc[has_tenure, "TotalCharges"]
    / df.loc[has_tenure, "tenure"]
)

# 5. Create senior citizen category
df["SeniorCitizenStatus"] = df["SeniorCitizen"].map({
    0: "Non-Senior",
    1: "Senior"
})

# 6. Create customer service category
df["ServiceCount"] = (
    df[[
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]]
    .eq("Yes")
    .sum(axis=1)
)

# 7. Display new features
print("\nNew Features Created:")
print(df[[
    "tenure",
    "TenureGroup",
    "AverageMonthlySpend",
    "SeniorCitizenStatus",
    "ServiceCount",
    "Churn"
]].head(10))

# 8. Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# 9. Save feature-engineered dataset
output_path = OUTPUT_DIR / "telco_features.csv"

df.to_csv(output_path, index=False)

print("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])
print("Saved File:", output_path)
