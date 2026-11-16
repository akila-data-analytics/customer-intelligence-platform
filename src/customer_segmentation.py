
# Customer Intelligence & Predictive Retention Platform - Customer Segmentation

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "telco_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_segments.csv"

# 2. Load cleaned dataset
df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)

# 3. Select customer features
features = ["tenure", "MonthlyCharges", "TotalCharges"]

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Required columns not found: {missing}")

X = df[features].apply(pd.to_numeric, errors="coerce")

# 4. Handle missing values
X = X.fillna(X.median())
X = X.fillna(0)

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Apply K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["CustomerSegment"] = kmeans.fit_predict(X_scaled)

# 7. Save customer segments
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# 8. Display segment counts
print("\nCustomer Segment Counts:")
print(df["CustomerSegment"].value_counts().sort_index())

# 9. Display segment profiles
profile_columns = features.copy()

# Convert churn into numeric form for calculating churn rate
if "Churn" in df.columns:
    df["ChurnFlag"] = (
        df["Churn"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0}
        )
    )
    profile_columns.append("ChurnFlag")

profiles = df.groupby("CustomerSegment")[profile_columns].mean().round(2)

if "ChurnFlag" in profiles.columns:
    profiles = profiles.rename(
        columns={"ChurnFlag": "ChurnRate"}
    )
    profiles["ChurnRate"] = (
        profiles["ChurnRate"] * 100
    ).round(2)

print("\nSegment Profiles:")
print(profiles)

print("\nSaved file:", OUTPUT_PATH)
print("Customer segmentation completed successfully!")
