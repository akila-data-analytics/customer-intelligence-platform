
# Customer Intelligence & Predictive Retention Platform - Dashboard

from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_segments.csv"

# 2. Dashboard settings
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Intelligence & Predictive Retention Platform")
st.write("Customer Churn Analysis and Segmentation Dashboard")

# 3. Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

# 4. Calculate KPIs
total_customers = len(df)
churned_customers = (df["Churn"].astype(str).str.strip().str.lower() == "yes").sum()
churn_rate = (
    churned_customers / total_customers * 100
    if total_customers > 0 else 0
)
average_monthly_charges = df["MonthlyCharges"].mean()

# 5. Display KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{churn_rate:.2f}%")
col4.metric("Average Monthly Charges", f"{average_monthly_charges:.2f}")

st.divider()

# 6. Customer segmentation chart
st.subheader("Customer Segment Distribution")

segment_counts = df["CustomerSegment"].value_counts().sort_index()

fig1, ax1 = plt.subplots()
ax1.bar(
    segment_counts.index.astype(str),
    segment_counts.values
)
ax1.set_xlabel("Customer Segment")
ax1.set_ylabel("Number of Customers")
ax1.set_title("Customers in Each Segment")

st.pyplot(fig1)
plt.close(fig1)

# 7. Segment profile
st.subheader("Customer Segment Profiles")

profile = df.groupby("CustomerSegment").agg(
    Customer_Count=("customerID", "count"),
    Average_Tenure=("tenure", "mean"),
    Average_Monthly_Charges=("MonthlyCharges", "mean"),
    Average_Total_Charges=("TotalCharges", "mean")
).round(2)

st.dataframe(profile, use_container_width=True)

# 8. Churn analysis by segment
st.subheader("Churn Rate by Customer Segment")

df["ChurnFlag"] = (
    df["Churn"].astype(str).str.strip().str.lower() == "yes"
).astype(int)

segment_churn = (
    df.groupby("CustomerSegment")["ChurnFlag"].mean() * 100
).round(2)

fig2, ax2 = plt.subplots()
ax2.bar(
    segment_churn.index.astype(str),
    segment_churn.values
)
ax2.set_xlabel("Customer Segment")
ax2.set_ylabel("Churn Rate (%)")
ax2.set_title("Churn Rate by Segment")

st.pyplot(fig2)
plt.close(fig2)


# 9. Automated Retention Recommendations

st.subheader("Automated Retention Recommendations")

for segment_id, row in profile.iterrows():

    churn_rate_value = segment_churn.get(segment_id, 0)
    avg_tenure = row["Average_Tenure"]
    avg_monthly = row["Average_Monthly_Charges"]

    st.markdown(f"### Customer Segment {segment_id}")

    st.write(f"Customers: {int(row['Customer_Count']):,}")
    st.write(f"Average tenure: {avg_tenure:.1f} months")
    st.write(f"Average monthly charges: {avg_monthly:.2f}")
    st.write(f"Observed churn rate: {churn_rate_value:.2f}%")

    if churn_rate_value >= 40:
        st.warning(
            "Action: Analyse churn reasons and prioritise "
            "customer feedback and retention outreach."
        )

    elif avg_tenure <= 12:
        st.info(
            "Action: Provide onboarding guidance, "
            "check customer satisfaction and explain useful services."
        )

    elif avg_monthly >= 80:
        st.info(
            "Action: Review service quality, billing concerns "
            "and whether customers receive good value."
        )

    else:
        st.success(
            "Action: Maintain service quality and monitor "
            "customer satisfaction regularly."
        )

st.divider()


# 9. Customer data preview
st.subheader("Customer Data Preview")
st.dataframe(df.head(20), use_container_width=True)

st.success("Dashboard loaded successfully!")
