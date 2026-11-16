
# Customer Intelligence & Predictive Retention Platform - Streamlit Dashboard

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.txt"
RECOMMENDATIONS_PATH = REPORTS_DIR / "retention_recommendations.csv"

# 2. Streamlit page configuration
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)


# 3. Load existing retention recommendations
@st.cache_data
def load_recommendations():
    if RECOMMENDATIONS_PATH.exists():
        df = pd.read_csv(RECOMMENDATIONS_PATH)

        if not df.empty:
            return df

    return None


# 4. Load trained model
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


# 5. Load saved decision threshold
@st.cache_data
def load_threshold():
    if THRESHOLD_PATH.exists():
        return float(
            THRESHOLD_PATH.read_text(encoding="utf-8").strip()
        )

    return 0.5


# 6. Load test data
@st.cache_data
def load_test_data():
    x_path = PROCESSED_DIR / "X_test.csv"
    y_path = PROCESSED_DIR / "y_test.csv"

    if not x_path.exists():
        return None, None

    X_test = pd.read_csv(x_path)

    y_test = None
    if y_path.exists():
        y_test = pd.read_csv(y_path).iloc[:, 0]

    return X_test, y_test


# 7. Create dashboard data
def create_dashboard_data():

    # First use the existing retention report
    df = load_recommendations()

    if df is not None:

        # Normalize existing risk labels
        risk_col = next(
            (
                column for column in df.columns
                if column.lower().replace(" ", "_") in [
                    "risk_level",
                    "risk_band",
                    "churn_risk",
                    "risk_category"
                ]
            ),
            None
        )

        probability_col = next(
            (
                column for column in df.columns
                if "probability" in column.lower()
                or "churn_prob" in column.lower()
            ),
            None
        )

        if risk_col:
            df["Risk_Level"] = (
                df[risk_col].astype(str).str.title()
            )

        elif probability_col:
            probability = pd.to_numeric(
                df[probability_col],
                errors="coerce"
            )

            # Support both decimal and percentage probabilities
            if probability.dropna().gt(1).any():
                probability = probability / 100

            df["Risk_Level"] = "Unknown"
            df.loc[probability < 0.30, "Risk_Level"] = "Low"
            df.loc[
                (probability >= 0.30) & (probability < 0.60),
                "Risk_Level"
            ] = "Medium"
            df.loc[probability >= 0.60, "Risk_Level"] = "High"

        if "Risk_Level" in df.columns:
            return df

    # Fallback: create predictions from test data
    model = load_model()
    X_test, y_test = load_test_data()
    threshold = load_threshold()

    if model is None or X_test is None:
        return None

    # Verify the model receives the expected columns
    model_columns = list(model.feature_names_in_)

    missing_columns = [
        column for column in model_columns
        if column not in X_test.columns
    ]

    if missing_columns:
        st.error(
            f"Test data is missing model features: {missing_columns}"
        )
        return None

    X_test = X_test[model_columns].copy()

    df = pd.DataFrame({
        "Test_Record_ID": range(1, len(X_test) + 1)
    })

    if y_test is not None:
        df["Actual_Churn"] = y_test.reset_index(drop=True)

    # Predict churn probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = list(model.classes_)

        if 1 not in classes:
            st.error("The model does not contain churn class 1.")
            return None

        churn_index = classes.index(1)
        churn_probability = probabilities[:, churn_index]

        df["Churn_Probability"] = churn_probability

        # Use the same saved threshold as the Flask application
        df["Predicted_Churn"] = (
            churn_probability >= threshold
        ).astype(int)

        # Dashboard risk bands are separate business reporting rules
        df["Risk_Level"] = "Low"
        df.loc[
            df["Churn_Probability"] >= 0.30,
            "Risk_Level"
        ] = "Medium"
        df.loc[
            df["Churn_Probability"] >= 0.60,
            "Risk_Level"
        ] = "High"

    else:
        df["Predicted_Churn"] = model.predict(X_test)
        df["Risk_Level"] = "Unknown"

    df["Retention_Recommendation"] = df["Risk_Level"].map({
        "High": "Prioritize customer outreach and retention review",
        "Medium": "Monitor customer activity and engagement",
        "Low": "Maintain service quality and satisfaction",
        "Unknown": "Review customer details"
    })

    return df


# 8. Dashboard heading
st.title("📊 Customer Intelligence & Predictive Retention Platform")

st.write(
    "Analyze customer churn risk, explore customer segments, "
    "and support customer retention decisions."
)

st.caption(
    "This dashboard displays the saved test-set retention report when "
    "available. Test record numbers are not actual customer IDs."
)

df = create_dashboard_data()

if df is None:
    st.error(
        "Dashboard data could not be loaded. Please check the model "
        "and processed CSV files in the project folders."
    )
    st.stop()

# Normalize risk labels
df["Risk_Level"] = (
    df["Risk_Level"].fillna("Unknown").astype(str).str.title()
)

# 9. KPI calculations
total_customers = len(df)

high_risk = (df["Risk_Level"] == "High").sum()
medium_risk = (df["Risk_Level"] == "Medium").sum()
low_risk = (df["Risk_Level"] == "Low").sum()
unknown_risk = (df["Risk_Level"] == "Unknown").sum()

# 10. KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{total_customers:,}")
col2.metric("High Risk", f"{high_risk:,}")
col3.metric("Medium Risk", f"{medium_risk:,}")
col4.metric("Low Risk", f"{low_risk:,}")

if unknown_risk:
    st.info(f"{unknown_risk:,} records have an unknown risk level.")

st.divider()

# 11. Risk distribution
st.subheader("Customer Churn Risk Distribution")

risk_counts = (
    df["Risk_Level"]
    .value_counts()
    .reindex(["High", "Medium", "Low", "Unknown"], fill_value=0)
)

left, right = st.columns(2)

with left:
    st.bar_chart(risk_counts)

with right:
    st.write("**Risk distribution summary**")

    summary = risk_counts.rename_axis(
        "Risk Level"
    ).reset_index(name="Records")

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

st.divider()

# 12. Customer filters
st.subheader("Customer Details")

risk_options = ["All"] + sorted(
    df["Risk_Level"].unique().tolist()
)

selected_risk = st.selectbox(
    "Filter by Risk Level",
    risk_options
)

filtered_df = df.copy()

if selected_risk != "All":
    filtered_df = filtered_df[
        filtered_df["Risk_Level"] == selected_risk
    ]

st.write(f"Records displayed: {len(filtered_df):,}")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# 13. Download filtered report
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Customer Report (CSV)",
    data=csv_data,
    file_name="customer_retention_report.csv",
    mime="text/csv"
)

st.divider()

# 14. Retention recommendations
st.subheader("Retention Recommendations")

if high_risk > 0:
    st.warning(
        f"{high_risk:,} records are classified as high risk. "
        "Review these records first and consider suitable retention support."
    )
else:
    st.success("No records are currently classified as high risk.")

st.write(
    "Medium risk: monitor engagement and service usage. "
    "Low risk: maintain service quality and customer satisfaction."
)

st.caption(
    "Risk bands use reporting cutoffs: High ≥ 60%, Medium ≥ 30% "
    "and below 60%, and Low below 30%. These bands are not guaranteed "
    "churn outcomes. They are separate from the model's saved prediction "
    "threshold. The dashboard may display a test-set report rather than "
    "live customer records."
)
