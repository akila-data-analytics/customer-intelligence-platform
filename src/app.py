
# Customer Intelligence & Predictive Retention Platform - Flask App

from flask import Flask, request, render_template_string
from pathlib import Path

import pandas as pd
import joblib

app = Flask(__name__)

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.txt"

# 2. Load trained model and threshold
model = joblib.load(MODEL_PATH)

threshold = float(
    THRESHOLD_PATH.read_text(encoding="utf-8").strip()
)

MODEL_COLUMNS = list(model.feature_names_in_)

print("Loaded model type:", type(model).__name__)
print("Model classes:", model.classes_)
print("Prediction threshold:", threshold)
print("Number of model features:", len(MODEL_COLUMNS))

# 3. Flask HTML page
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f5f9;
            padding: 20px;
            margin: 0;
        }

        main {
            max-width: 850px;
            margin: auto;
            background: white;
            padding: 25px;
            border-radius: 12px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
        }

        input, select {
            width: 100%;
            padding: 10px;
            box-sizing: border-box;
        }

        button {
            margin-top: 20px;
            padding: 12px 20px;
            cursor: pointer;
            background: #175cd3;
            color: white;
            border: none;
            border-radius: 6px;
        }

        .result {
            margin-top: 20px;
            padding: 15px;
            background: #eaf4ff;
            border-radius: 8px;
        }

        .error {
            margin-top: 20px;
            color: darkred;
        }

        @media (max-width: 600px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
<main>
    <h1>Customer Churn Prediction</h1>
    <p>Enter customer details to estimate churn risk.</p>

    <form method="POST">
        <div class="grid">

        {% for field, options in fields.items() %}
            <div>
                <label>{{ field }}</label>

                <select name="{{ field }}" required>
                {% for value in options %}
                    <option value="{{ value }}"
                    {% if values.get(field, options[0]) == value %}selected{% endif %}>
                        {{ value }}
                    </option>
                {% endfor %}
                </select>
            </div>
        {% endfor %}

            <div>
                <label>Tenure (months)</label>
                <input
                    type="number"
                    name="tenure"
                    min="0"
                    max="72"
                    value="{{ values.get('tenure', '12') }}"
                    required
                >
            </div>

            <div>
                <label>Monthly Charges</label>
                <input
                    type="number"
                    name="MonthlyCharges"
                    min="0"
                    step="0.01"
                    value="{{ values.get('MonthlyCharges', '70') }}"
                    required
                >
            </div>

            <div>
                <label>Total Charges</label>
                <input
                    type="number"
                    name="TotalCharges"
                    min="0"
                    step="0.01"
                    value="{{ values.get('TotalCharges', '840') }}"
                    required
                >
            </div>

        </div>

        <button type="submit">Predict Churn</button>
    </form>

    {% if result %}
    <section class="result">
        <h2>{{ result }}</h2>
        <p>Estimated churn probability: {{ probability }}%</p>
        <p>Prediction threshold: {{ threshold_percent }}%</p>
        <p>{{ message }}</p>
    </section>
    {% endif %}

    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}

</main>
</body>
</html>
"""

# 4. Customer input fields
FIELDS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}


# 5. Convert form data into model features
def build_features(form):

    try:
        tenure = int(form["tenure"])
        monthly = float(form["MonthlyCharges"])
        total = float(form["TotalCharges"])
        senior = int(form["SeniorCitizen"])
    except (ValueError, TypeError, KeyError):
        raise ValueError("Please enter valid numeric customer details.")

    if not 0 <= tenure <= 72:
        raise ValueError("Tenure must be between 0 and 72 months.")

    if monthly < 0 or total < 0:
        raise ValueError("Charges cannot be negative.")

    if senior not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1.")

    if not all(map(lambda value: pd.notna(value), [monthly, total])):
        raise ValueError("Charges must be valid numbers.")

    # Match tenure groups used during feature engineering
    if tenure <= 12:
        tenure_group = "New Customer"
    elif tenure <= 24:
        tenure_group = "1-2 Years"
    elif tenure <= 48:
        tenure_group = "2-4 Years"
    else:
        tenure_group = "4-6 Years"

    # Match the service-count logic used during training
    service_fields = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    service_count = sum(
        form[name] == "Yes" for name in service_fields
    )

    # Build all model features explicitly
    features = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "AverageMonthlySpend": total / tenure if tenure > 0 else 0.0,
        "ServiceCount": service_count,

        "gender_Male": int(form["gender"] == "Male"),
        "Partner_Yes": int(form["Partner"] == "Yes"),
        "Dependents_Yes": int(form["Dependents"] == "Yes"),
        "PhoneService_Yes": int(form["PhoneService"] == "Yes"),

        "MultipleLines_No phone service": int(
            form["MultipleLines"] == "No phone service"
        ),
        "MultipleLines_Yes": int(form["MultipleLines"] == "Yes"),

        "InternetService_Fiber optic": int(
            form["InternetService"] == "Fiber optic"
        ),
        "InternetService_No": int(form["InternetService"] == "No"),

        "OnlineSecurity_No internet service": int(
            form["OnlineSecurity"] == "No internet service"
        ),
        "OnlineSecurity_Yes": int(form["OnlineSecurity"] == "Yes"),

        "OnlineBackup_No internet service": int(
            form["OnlineBackup"] == "No internet service"
        ),
        "OnlineBackup_Yes": int(form["OnlineBackup"] == "Yes"),

        "DeviceProtection_No internet service": int(
            form["DeviceProtection"] == "No internet service"
        ),
        "DeviceProtection_Yes": int(form["DeviceProtection"] == "Yes"),

        "TechSupport_No internet service": int(
            form["TechSupport"] == "No internet service"
        ),
        "TechSupport_Yes": int(form["TechSupport"] == "Yes"),

        "StreamingTV_No internet service": int(
            form["StreamingTV"] == "No internet service"
        ),
        "StreamingTV_Yes": int(form["StreamingTV"] == "Yes"),

        "StreamingMovies_No internet service": int(
            form["StreamingMovies"] == "No internet service"
        ),
        "StreamingMovies_Yes": int(form["StreamingMovies"] == "Yes"),

        "Contract_One year": int(form["Contract"] == "One year"),
        "Contract_Two year": int(form["Contract"] == "Two year"),

        "PaperlessBilling_Yes": int(
            form["PaperlessBilling"] == "Yes"
        ),

        "PaymentMethod_Credit card (automatic)": int(
            form["PaymentMethod"] == "Credit card (automatic)"
        ),
        "PaymentMethod_Electronic check": int(
            form["PaymentMethod"] == "Electronic check"
        ),
        "PaymentMethod_Mailed check": int(
            form["PaymentMethod"] == "Mailed check"
        ),

        "TenureGroup_2-4 Years": int(
            tenure_group == "2-4 Years"
        ),
        "TenureGroup_4-6 Years": int(
            tenure_group == "4-6 Years"
        ),
        "TenureGroup_New Customer": int(
            tenure_group == "New Customer"
        ),

        "SeniorCitizenStatus_Senior": int(senior == 1)
    }

    # Verify exact feature names and order
    missing = [col for col in MODEL_COLUMNS if col not in features]
    extra = [col for col in features if col not in MODEL_COLUMNS]

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Missing: {missing}; Extra: {extra}"
        )

    X = pd.DataFrame(
        [[features[col] for col in MODEL_COLUMNS]],
        columns=MODEL_COLUMNS
    )

    print("Model input shape:", X.shape)

    return X


# 6. Prediction page
@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probability = None
    message = ""
    error = None

    values = request.form if request.method == "POST" else {}

    if request.method == "POST":
        try:
            X = build_features(request.form)

            # Use the saved threshold selected during model training
            churn_probability = float(
                model.predict_proba(X)[0][1]
            )

            prediction = int(churn_probability >= threshold)
            probability = round(churn_probability * 100, 2)

            print("Predicted class:", prediction)
            print("Churn probability:", probability, "%")

            if prediction == 1:
                result = "Model predicts this customer may churn."
                message = (
                    "Review customer needs and consider suitable "
                    "retention support."
                )
            else:
                result = "Model predicts this customer is likely to stay."
                message = (
                    "Continue good service and monitor "
                    "customer satisfaction."
                )

        except (ValueError, KeyError) as exc:
            error = str(exc)
            app.logger.warning("Invalid prediction input: %s", exc)

        except Exception:
            error = (
                "Prediction could not be completed. "
                "Please check the input values and try again."
            )
            app.logger.exception("Prediction failed")

    return render_template_string(
        HTML,
        fields=FIELDS,
        values=values,
        result=result,
        probability=probability,
        threshold_percent=round(threshold * 100, 2),
        message=message,
        error=error
    )


# 7. Start Flask application
if __name__ == "__main__":
    app.run(debug=True)
