
# Exploratory Data Analysis

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("DATASET LOADED SUCCESSFULLY")
print("-" * 50)

# 3. Basic dataset information
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# 4. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 6. Clean TotalCharges column
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

print("\nTotalCharges converted to numeric.")

# 7. Churn analysis
print("\nChurn Count:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
churn_percentage = df["Churn"].value_counts(normalize=True) * 100
print(churn_percentage.round(2))

# 8. Save churn summary
churn_summary = pd.DataFrame({
    "Customer_Status": churn_percentage.index,
    "Percentage": churn_percentage.values
})

churn_summary.to_csv(
    OUTPUT_DIR / "churn_summary.csv", index=False
)

# 9. Numerical analysis
print("\nNumerical Summary:")
print(df.describe())

# 10. Churn by contract type
print("\nChurn by Contract:")
contract_churn = pd.crosstab(
    df["Contract"], df["Churn"], normalize="index"
) * 100

print(contract_churn.round(2))

# 11. Create charts

# Chart 1: Churn Count
df["Churn"].value_counts().plot(
    kind="bar", title="Customer Churn Count"
)
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_count.png")
plt.show()
plt.close()

# Chart 2: Tenure by Churn
df.boxplot(column="tenure", by="Churn")
plt.title("Customer Tenure by Churn")
plt.suptitle("")
plt.xlabel("Churn Status")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tenure_by_churn.png")
plt.show()
plt.close()

# Chart 3: Monthly Charges by Churn
df.boxplot(column="MonthlyCharges", by="Churn")
plt.title("Monthly Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_charges_by_churn.png")
plt.show()
plt.close()

# Chart 4: Contract Type vs Churn
contract_churn.plot(
    kind="bar",
    title="Churn Percentage by Contract Type"
)
plt.xlabel("Contract Type")
plt.ylabel("Customers (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "contract_churn.png")
plt.show()
plt.close()

# 12. Save EDA report
report_path = OUTPUT_DIR / "eda_summary.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("CUSTOMER INTELLIGENCE & PREDICTIVE RETENTION PLATFORM\n")
    file.write("AUTOMATED EDA REPORT\n\n")
    file.write(f"Dataset rows: {df.shape[0]}\n")
    file.write(f"Dataset columns: {df.shape[1]}\n")
    file.write(f"Duplicate records: {df.duplicated().sum()}\n")
    file.write("\nMissing Values:\n")
    file.write(df.isnull().sum().to_string())
    file.write("\n\nChurn Percentage:\n")
    file.write(churn_percentage.round(2).to_string())
    file.write("\n\nNumerical Summary:\n")
    file.write(df.describe().to_string())
    file.write("\n\nChurn by Contract:\n")
    file.write(contract_churn.round(2).to_string())

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Reports saved in:", OUTPUT_DIR)
