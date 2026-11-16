
# Customer Churn Prediction

import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/best_churn_model.joblib")

# Load training columns to maintain the correct feature order
X_train = pd.read_csv("data/processed/X_train.csv")
feature_columns = X_train.columns

# Get customer details
print("\nCUSTOMER CHURN PREDICTION")
print("-" * 35)

tenure = int(input("Customer tenure in months (0-72): "))
monthly_charges = float(input("Monthly charges: "))
total_charges = float(input("Total charges: "))
contract = input("Contract (Month-to-month / One year / Two year): ")
internet = input("Internet service (DSL / Fiber optic / No): ")
payment = input("Payment method (Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)): ")

# Validate basic inputs
if not 0 <= tenure <= 72:
    raise ValueError("Tenure must be between 0 and 72 months.")

if monthly_charges < 0 or total_charges < 0:
    raise ValueError("Charges cannot be negative.")

# Build customer record using original feature names
customer = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "AverageMonthlySpend": total_charges / tenure if tenure > 0 else 0,
    "SeniorCitizen": 0,
    "SeniorCitizenStatus": "Non-Senior",
    "gender": "Male",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
    "TenureGroup": (
        "New Customer" if tenure <= 12
        else "Loyal Customer" if tenure <= 48
        else "Long-term Customer"
    ),
    "ServiceCount": 1,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment
}

# Convert customer details to a DataFrame
customer_df = pd.DataFrame([customer])

# Convert categorical values into model features
customer_df = pd.get_dummies(
    customer_df,
    drop_first=True,
    dtype=int
)

# Match the model's training columns
customer_df = customer_df.reindex(
    columns=feature_columns,
    fill_value=0
)

# Predict churn
prediction = model.predict(customer_df)[0]
probability = model.predict_proba(customer_df)[0][1]

# Display result
print("\n" + "=" * 40)
print("CUSTOMER CHURN PREDICTION RESULT")
print("=" * 40)

if prediction == 1:
    print("Prediction: Customer may CHURN")
else:
    print("Prediction: Customer is likely to STAY")

print(f"Churn probability: {probability * 100:.2f}%")

# Retention suggestion
if probability >= 0.60:
    print("Recommendation: Contact customer and offer suitable retention support.")
elif probability >= 0.30:
    print("Recommendation: Monitor customer satisfaction and service usage.")
else:
    print("Recommendation: Continue regular customer engagement.")

print("=" * 40)
