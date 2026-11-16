# Customer Intelligence & Predictive Retention Platform

## 1. Project Overview

This project analyzes customer data to understand customer behavior, identify customers who may leave a service, and support customer retention decisions using machine learning.

The platform includes data validation, preprocessing, exploratory data analysis, feature engineering, model comparison, churn prediction, customer segmentation, retention recommendations, a Flask prediction application, and a Streamlit dashboard.

## 2. Project Objectives

- Analyze customer churn patterns.
- Validate and clean customer data.
- Perform exploratory data analysis (EDA).
- Engineer features for machine learning.
- Compare multiple classification models against a baseline.
- Evaluate models using metrics appropriate for class imbalance.
- Predict customer churn using a saved model and selected threshold.
- Segment customers using K-Means clustering.
- Generate risk categories and retention recommendations.
- Display insights through a Streamlit dashboard.
- Provide individual customer predictions through a Flask web application.
- Generate visualizations and full project documentation.

## 3. Dataset

**Dataset:** IBM Telco Customer Churn

**Original file:** `data/raw/Telco-Customer-Churn.csv`

The dataset contains 7,043 customer records and 21 original columns. It includes customer tenure, monthly charges, total charges, contract details, services, payment methods, and churn status.

The target variable is `Churn`, which indicates whether a customer left the service.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Flask
- Streamlit
- Pytest
- ReportLab

## 5. Project Workflow

1. Validate the original dataset.
2. Clean missing and inconsistent values.
3. Perform exploratory data analysis.
4. Create additional features.
5. Prepare training and test datasets.
6. Compare classification models using validation data.
7. Select a model and decision threshold.
8. Evaluate model performance on held-out test data.
9. Analyze feature importance.
10. Segment customers using K-Means.
11. Generate churn-risk categories and retention recommendations.
12. Create visualizations and documentation.
13. Serve predictions through Flask and display insights through Streamlit.
14. Run automated model tests.

## 6. Project Modules

All Python modules are located in the `src/` directory.

| File | Purpose |
|---|---|
| `data_validation.py` | Checks dataset dimensions, duplicates, target distribution, and data quality. |
| `data_cleaning.py` | Cleans data and handles missing values. |
| `automated_eda.py` | Performs exploratory data analysis. |
| `feature_engineering.py` | Creates features such as tenure group, average monthly spend, senior-citizen status, and service count. |
| `prepare_ml_data.py` | Encodes features and prepares training and test datasets. |
| `train_model.py` | Compares a baseline with multiple classification models, selects a model and threshold, and saves model artifacts. |
| `evaluate_model.py` | Evaluates the selected model on held-out test data. |
| `feature_importance.py` | Produces a feature-importance analysis for the selected model. |
| `predict_churn.py` | Provides a prediction-related script. |
| `customer_segmentation.py` | Groups customers using K-Means clustering and summarizes segment profiles. |
| `retention_recommendations.py` | Generates risk categories and rule-based retention recommendations. |
| `save_visualizations.py` | Generates project charts and saves them in the reports directory. |
| `create_visualization_pdf.py` | Converts Markdown documentation into a PDF and includes available visualization charts. |
| `app.py` | Flask web application for individual customer churn predictions. |
| `streamlit_app.py` | Streamlit dashboard for customer-risk and retention insights. |
| `dashboard.py` | Additional dashboard-related module. |

## 7. Model Training and Evaluation

The training workflow compares the following classifiers:

- Dummy Classifier baseline
- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosting

The selected model is saved as:

`models/best_churn_model.joblib`

The selected decision threshold is saved as:

`models/best_threshold.txt`

Model selection uses validation F1-score. The selected threshold is determined from validation predictions to maximize F1-score. The final evaluation is performed on the held-out test set.

### Selected Model Results

The current saved evaluation results for Logistic Regression are:

| Metric | Test Result |
|---|---:|
| Accuracy | 0.7850 |
| Precision | 0.5872 |
| Recall | 0.6390 |
| F1-score | 0.6120 |
| ROC-AUC | 0.8421 |
| PR-AUC (Average Precision) | 0.6365 |

These results describe performance on the current test split. They do not guarantee the same performance on new customers or future data.

Accuracy is reported alongside precision, recall, F1-score, ROC-AUC, and PR-AUC because the dataset has an imbalanced churn distribution.

## 8. Customer Segmentation

K-Means clustering is used to group customers based on tenure, monthly charges, and total charges.

The current segment profiles indicate:

- **Segment 0:** Lower average monthly charges and a relatively low churn rate.
- **Segment 1:** Longer tenure and higher average monthly charges, with a relatively low churn rate.
- **Segment 2:** Shorter tenure, relatively high monthly charges, and the highest churn rate among the three segments.

Segment numbers are arbitrary cluster labels, not customer rankings. Segment profiles and churn rates should be interpreted as descriptive findings from this dataset.

## 9. Risk Categories and Retention Recommendations

The retention report groups customers using these reporting rules:

- **High risk:** Churn probability greater than or equal to 0.60.
- **Medium risk:** Churn probability greater than or equal to 0.30 and below 0.60.
- **Low risk:** Churn probability below 0.30.

These risk-category cutoffs are dashboard/reporting rules. They are separate from the selected model's prediction threshold, which is approximately 0.6565 in the current saved model.

Recommendations are rule-based suggestions and should be reviewed by the business before being used in customer communications.

The generated report uses `Test_Record_ID` to identify rows in the test dataset. It is not the original customer ID and cannot independently identify or contact a real customer.

## 10. Applications

### Flask Prediction Application

From the project root directory, run:

```bash
python src/app.py
```

Open the following address in a browser:

`http://127.0.0.1:5000`

Enter customer information in the form to receive a churn probability and a prediction based on the saved threshold.

The form's initial values are sample defaults for demonstration. They are not actual customer records.

### Streamlit Dashboard

Open another terminal from the project root directory and run:

```bash
python -m streamlit run src/streamlit_app.py
```

The dashboard displays customer-risk summaries, risk categories, and retention recommendations based on the available project outputs.

The Flask development server is intended for local development and testing, not production deployment.

## 11. How to Set Up the Project

Use Python 3.12 or another compatible Python environment.

Install the dependencies from the project root directory:

```bash
python -m pip install -r requirements.txt
```

The original dataset should be available at:

`data/raw/Telco-Customer-Churn.csv`

### Run the Main Workflow

Run these commands from the project root directory when you need to reproduce the workflow:

```bash
python src/data_validation.py
python src/data_cleaning.py
python src/automated_eda.py
python src/feature_engineering.py
python src/prepare_ml_data.py
python src/train_model.py
python src/evaluate_model.py
python src/feature_importance.py
python src/customer_segmentation.py
python src/retention_recommendations.py
python src/save_visualizations.py
python src/create_visualization_pdf.py
```

Run the workflow in order because later steps depend on outputs from earlier steps. If the training data or model is regenerated, rerun the dependent evaluation and reporting steps.

## 12. Automated Tests

Run the model tests using:

```bash
python -m pytest tests/test_model.py -v
```

The current test suite contains four tests that check:

- Whether the saved model file exists.
- Whether the model loads successfully.
- Whether the model has a prediction method.
- Whether the model generates a prediction for a sample input.

The current test run passed all four tests. These tests check basic model availability and prediction functionality; they do not establish model accuracy or production readiness.

## 13. Reports and Documentation

The `reports/` directory contains generated evaluation results, charts, recommendations, and documentation.

Important outputs include:

- `model_evaluation_metrics.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `feature_importance.csv`
- `feature_importance.png`
- `retention_recommendations.csv`
- `Customer_Churn_Full_Documentation.md`
- `Customer_Churn_Full_Documentation.pdf`

The visualization script generates 15 charts, including churn patterns, model comparison, customer segments, risk-category counts, and monthly charges associated with risk groups.

## 14. Limitations and Assumptions

- The project uses a historical Telco Customer Churn dataset. Its findings may not generalize to every business or future customer population.
- The dataset contains 11 `TotalCharges` values that become missing when converted to numeric values. Zero-tenure records are handled using zero total charges; any remaining missing values are filled using `MonthlyCharges`. This is a preprocessing assumption and may not represent the actual billing history.
- Model selection and threshold tuning use validation data, while the reported final metrics use the held-out test set. The selected threshold maximizes validation F1-score; it has not been proven to maximize business profit or minimize retention costs.
- Churn predictions are estimates, not guaranteed outcomes.
- Risk categories use reporting cutoffs and should not be interpreted as calibrated business decisions without further validation.
- Feature importance for Logistic Regression is based on absolute coefficient magnitudes. These magnitudes show association strength, not direction, and can be affected by differences in feature scale.
- Customer segments are descriptive groups. Cluster numbers do not imply an inherent ranking or causal explanation.
- Monthly charges grouped by risk category are an exposure indicator, not measured revenue loss or guaranteed recoverable revenue.
- `Test_Record_ID` represents a row in the test dataset, not the original customer ID. A reliable mapping is needed before operational customer outreach.
- Retention recommendations are rule-based suggestions and require business validation.
- The Flask development server is not configured for production deployment.
- The current automated tests cover basic model functionality. More tests are needed for data preprocessing, threshold behavior, API inputs, and dashboard workflows.

## 15. Future Improvements

- Expand automated test coverage.
- Validate and improve missing-value handling.
- Evaluate threshold choices using retention costs, customer value, and business constraints.
- Improve feature preprocessing and monitor model performance.
- Track model versions and data changes.
- Connect predictions to a verified customer identifier when an appropriate data source is available.
- Validate retention recommendations using business feedback.
- Deploy the application using an appropriate production configuration.

## 16. Project Folder Structure

```text
customer-intelligence-platform/
|
|-- data/
|   |-- raw/
|   |   |-- Telco-Customer-Churn.csv
|   |
|   |-- processed/
|       |-- telco_cleaned.csv
|       |-- telco_features.csv
|       |-- customer_segments.csv
|       |-- model_comparison.csv
|       |-- X_train.csv
|       |-- X_test.csv
|       |-- y_train.csv
|       |-- y_test.csv
|
|-- notebooks/
|   |-- 01_customer_churn_eda.ipynb
|
|-- src/
|   |-- app.py
|   |-- streamlit_app.py
|   |-- dashboard.py
|   |-- data_validation.py
|   |-- data_cleaning.py
|   |-- automated_eda.py
|   |-- feature_engineering.py
|   |-- prepare_ml_data.py
|   |-- train_model.py
|   |-- evaluate_model.py
|   |-- feature_importance.py
|   |-- predict_churn.py
|   |-- customer_segmentation.py
|   |-- retention_recommendations.py
|   |-- save_visualizations.py
|   |-- create_visualization_pdf.py
|
|-- models/
|   |-- best_churn_model.joblib
|   |-- best_threshold.txt
|
|-- reports/
|   |-- Generated charts and analysis reports
|   |-- Customer_Churn_Full_Documentation.md
|   |-- Customer_Churn_Full_Documentation.pdf
|
|-- tests/
|   |-- test_model.py
|
|-- README.md
|-- requirements.txt
```

## 17. Conclusion

The Customer Intelligence & Predictive Retention Platform demonstrates a complete customer-churn analytics workflow, from data preparation and model evaluation to customer segmentation, risk reporting, and prediction applications. The outputs can support further retention analysis, but real-world use requires business validation, reliable customer identification, and ongoing model monitoring.