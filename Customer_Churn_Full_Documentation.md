# Customer Intelligence & Predictive Retention Platform

## 1. Project Introduction

The Customer Intelligence & Predictive Retention Platform is a machine learning-based project designed to analyze customer behavior, identify customers who may leave a service, and support customer retention strategies.

Customer churn can affect business revenue and long-term customer relationships. By analyzing historical customer information, businesses can identify churn patterns and take preventive actions.

This project combines data analysis, machine learning, customer segmentation, data visualization, and a web-based prediction application.

## 2. Problem Statement

Businesses may find it difficult to identify customers who are likely to discontinue their services. Without analyzing customer behavior, companies may lose customers and miss opportunities to improve customer satisfaction.

This project uses historical customer data to study churn patterns, predict potential churn, segment customers, and generate retention recommendations.

## 3. Project Objectives

- Validate and clean the customer dataset.
- Perform Exploratory Data Analysis (EDA).
- Identify patterns associated with customer churn.
- Prepare data for machine learning.
- Train and evaluate classification models.
- Predict customer churn using a trained model.
- Segment customers based on their characteristics.
- Categorize customers by predicted churn risk.
- Generate customer retention recommendations.
- Develop a Flask prediction application.
- Build a Streamlit analytics dashboard.
- Test the trained model and document project results.

## 4. Dataset Description

### 4.1 Dataset Source

This project uses the IBM Telco Customer Churn dataset, which contains telecommunications customer information, subscribed services, contract details, billing information, and churn status.

### 4.2 Dataset Overview

| Attribute | Description |
|---|---|
| Dataset | IBM Telco Customer Churn |
| Original records | 7,043 |
| Original columns | 21 |
| Data format | Structured tabular data |
| Target variable | Churn |
| Main purpose | Churn analysis and prediction |

### 4.3 Important Features

- **customerID:** Unique customer identifier.
- **gender:** Gender recorded in the dataset.
- **SeniorCitizen:** Indicates senior-citizen status.
- **Partner:** Whether the customer has a partner.
- **Dependents:** Whether the customer has dependents.
- **tenure:** Number of months the customer has stayed.
- **InternetService:** Type of internet service.
- **Contract:** Customer contract type.
- **PaymentMethod:** Customer payment method.
- **MonthlyCharges:** Monthly service charges.
- **TotalCharges:** Total charges recorded for the customer.
- **Churn:** Indicates whether the customer left the service.

The `Churn` column is the target variable for supervised machine learning.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading, cleaning, and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical charts and heatmaps |
| Scikit-learn | Machine learning and evaluation |
| Joblib | Saving and loading the trained model |
| Flask | Web-based prediction application |
| Streamlit | Interactive analytics dashboard |
| Pytest | Automated testing |
| Jupyter Notebook | Exploratory analysis |
| Git/GitHub | Version control, if configured |

## 6. Project Workflow

1. Load and validate the original customer dataset.
2. Clean data and prepare analysis-ready records.
3. Perform Exploratory Data Analysis.
4. Create visualizations to understand customer behavior.
5. Engineer features and prepare training and testing datasets.
6. Train and compare machine learning models.
7. Evaluate model performance using appropriate metrics.
8. Perform customer segmentation and churn-risk analysis.
9. Generate customer retention recommendations.
10. Integrate the trained model into a Flask application.
11. Present analysis through a Streamlit dashboard.
12. Run automated tests and prepare project documentation.

## 7. Data Validation and Cleaning

### 7.1 Data Validation

Data validation was performed to check the structure and quality of the original dataset before analysis.

The original IBM Telco Customer Churn dataset contained 7,043 records and 21 columns.

The validation process included checking:
- Dataset dimensions and column names.
- Data types of important columns.
- Missing or incomplete values.
- Data quality issues that could affect analysis.

### 7.2 Data Cleaning

Data cleaning was performed to prepare the customer dataset for reliable analysis and machine learning.

The cleaning process included:
- Inspecting missing values.
- Checking and converting data types where required.
- Preparing numerical and categorical features.
- Saving the cleaned dataset for subsequent analysis.

The cleaned dataset was saved as `data/processed/telco_cleaned.csv`.

### 7.3 Feature Engineering

Feature engineering prepares useful input variables for machine learning.

The project generated additional model-ready features from customer information. Categorical variables and numerical variables were prepared for model training through the project's preprocessing workflow.

The prepared feature dataset was saved as `data/processed/telco_features.csv`.

## 8. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the distribution of customer data and investigate patterns associated with churn.

The analysis examined customer churn status, contract types, internet service plans, tenure, monthly charges, and relationships between numerical variables.

The EDA process helped identify patterns that could support model development and business investigation.

### 8.1 Churn Distribution

The churn distribution chart compares the number of customers who left the service with the number who remained.

**Purpose:** Understand the balance between churned and retained customers.

**Business use:** Estimate the scale of customer churn and identify the need for retention initiatives.

### 8.2 Contract-Based Churn Analysis

The contract churn chart compares churn rates across contract types.

**Purpose:** Investigate whether churn rates differ between contract groups.

**Business use:** Identify contract categories that may require further investigation or targeted retention strategies.

### 8.3 Internet Plan Churn Analysis

The internet plan chart compares churn rates across internet service categories.

**Purpose:** Understand how churn varies between service-plan groups.

**Business use:** Investigate whether customer experience, pricing, or service differences may be associated with churn.

### 8.4 Monthly Charges Analysis

The monthly charges box plot compares customer charges according to churn status.

**Purpose:** Examine differences in monthly charges between churned and retained customers.

**Business use:** Explore whether pricing levels may be associated with customer churn.

### 8.5 Tenure Analysis

The tenure box plot compares how long churned and retained customers have stayed with the company.

**Purpose:** Understand the relationship between customer tenure and churn status.

**Business use:** Consider early-stage customer support and onboarding strategies where appropriate.

### 8.6 Correlation Analysis

A numerical correlation heatmap was generated to examine linear relationships between numerical features.

**Purpose:** Identify numerical variables that move together and may deserve further analysis.

**Business use:** Support feature investigation and improve understanding of customer data.

**Important note:** Correlation does not prove that one variable causes another variable to change.

## 9. Machine Learning Model Development

### 9.1 Data Preparation for Machine Learning

The cleaned customer dataset was prepared for supervised machine learning.

The target variable was `Churn`, which indicates whether a customer left the service.

The data preparation workflow included:
- Separating input features and the target variable.
- Preparing numerical and categorical features.
- Splitting the dataset into training and testing sets.
- Applying the required preprocessing steps before model training.

The prepared datasets were saved in the `data/processed/` directory.

### 9.2 Models Trained

Four classification approaches and a baseline model were evaluated in the project.

**Logistic Regression**

Logistic Regression estimates the probability that a customer belongs to a particular class. It provides a useful classification model and achieved the highest accuracy and F1-score among the evaluated models.

**Random Forest**

Random Forest combines predictions from multiple decision trees. It achieved the highest recall among the four trained classification models, helping identify a larger proportion of actual churners.

**Gradient Boosting**

Gradient Boosting builds an ensemble of models sequentially, with each stage working to improve the previous stages. It achieved the highest Precision-Recall AUC (PR-AUC) among the evaluated classification models.

**Decision Tree**

Decision Tree uses a sequence of feature-based decisions to classify customers. Its performance was lower than the Logistic Regression and ensemble models in this evaluation.

**Baseline Model**

The baseline provides a reference point for comparing the trained models. Its results help demonstrate why a model must be evaluated using more than accuracy alone.

### 9.3 Model Comparison

The model comparison results are stored in:

`data/processed/model_comparison.csv`

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Classification threshold

The recorded results are shown below. Metrics are rounded to three decimal places.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.784 | 0.583 | 0.647 | 0.613 | 0.842 | 0.636 |
| Random Forest | 0.754 | 0.526 | 0.727 | 0.611 | 0.824 | 0.616 |
| Gradient Boosting | 0.761 | 0.538 | 0.703 | 0.610 | 0.842 | 0.657 |
| Decision Tree | 0.739 | 0.508 | 0.505 | 0.507 | 0.664 | 0.388 |
| Baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |

Logistic Regression achieved the highest accuracy and F1-score among the trained classification models. Random Forest achieved the highest recall, while Gradient Boosting achieved the highest PR-AUC.

These results demonstrate that model performance depends on the metric being considered. For churn prediction, recall and PR-AUC are particularly useful because the business needs to identify potential churners in a dataset where churned and retained customers are not equally represented.

### 9.4 Model Selection

The selected model for the current prediction workflow is **Gradient Boosting Classifier**.

The selected model is saved as:

`models/best_churn_model.joblib`

The prediction threshold recorded for Gradient Boosting is approximately **0.3423**. This threshold is used to convert estimated churn probabilities into class predictions in the evaluated workflow.

Gradient Boosting was selected with consideration for its PR-AUC of approximately 0.657, the highest among the evaluated models, and its ability to identify potential churners. This does not mean that it performs best on every metric.

The saved model is loaded by the prediction application to generate churn predictions. Model selection should balance the identification of actual churners against unnecessary retention alerts and should be reviewed as new customer data becomes available.

## 10. Model Evaluation

Model evaluation measures how well the trained model performs on testing data that was not used to fit the model.

### 10.1 Accuracy

Accuracy measures the proportion of all predictions that are correct.

### 10.2 Precision

Precision measures the proportion of customers predicted as churners who actually churned.

Higher precision can help reduce unnecessary retention outreach.

### 10.3 Recall

Recall measures the proportion of actual churners correctly identified by the model.

Higher recall can help the business identify more customers who may leave.

### 10.4 F1-score

The F1-score combines precision and recall into a single metric. It is useful when both missed churners and false alerts matter.

### 10.5 ROC-AUC

ROC-AUC measures how well a model distinguishes between churned and retained customers across different classification thresholds.

### 10.6 Confusion Matrix

The confusion matrix summarizes correct and incorrect predictions.

- **True Positive:** A churner correctly predicted as a churner.
- **True Negative:** A retained customer correctly predicted as retained.
- **False Positive:** A retained customer incorrectly predicted as a churner.
- **False Negative:** A churner incorrectly predicted as retained.

False negatives are especially important because customers at risk of leaving may not receive timely retention attention.

### 10.7 Evaluation Limitations

Model metrics describe performance on the evaluation dataset and may not represent future performance exactly.

The model should be evaluated periodically using new customer data before its predictions are relied upon for operational decisions.

## 11. Customer Segmentation

Customer segmentation is the process of dividing customers into groups based on shared characteristics or behavior.

In this project, customer segmentation was performed to help understand differences between customer groups and support targeted retention strategies.

### 11.1 Segmentation Process

The project generated customer segment assignments and saved them in:

`data/processed/customer_segments.csv`

The segmentation output includes a `CustomerSegment` column that identifies the assigned segment for each customer.

### 11.2 Segment Visualization

A two-dimensional scatter plot was created using customer tenure and monthly charges, with different colors representing the assigned segments.

This visualization helps explore how customer groups are distributed across these two characteristics.

### 11.3 Business Benefits

Customer segmentation can help a business:
- Understand differences in customer behavior.
- Identify groups that may need additional support.
- Design more relevant customer communication.
- Develop retention strategies for different customer groups.

**Note:** The scatter plot shows only two dimensions. It does not represent every feature used to define a segment.

## 12. Churn Risk Prediction

The trained machine learning model generates churn predictions and estimated churn probabilities for customers.

### 12.1 Churn Probability

A churn probability is the model's estimated likelihood that a customer belongs to the churn class. It is not a guarantee that the customer will leave.

### 12.2 Risk Categories

For the project's risk analysis, customers were grouped using these probability thresholds:

| Risk Category | Predicted Churn Probability |
|---|---|
| Low Risk | Up to 30% |
| Medium Risk | Above 30% and up to 60% |
| High Risk | Above 60% |

These are project-defined thresholds used to organize predictions into risk groups.

### 12.3 Risk Category Analysis

The risk category chart shows the number of test customers in each risk group.

The churn probability distribution chart shows how predicted probabilities are spread across the testing dataset.

These charts can help prioritize customers for further review.

### 12.4 Revenue Exposure Analysis

The project also summarizes the monthly charges associated with customers in each risk category.

This analysis helps identify the amount of monthly billing associated with different predicted risk groups.

**Important limitation:** Monthly charges associated with high-risk customers are not the same as actual revenue loss. The analysis does not establish that all high-risk customers will leave or that all their monthly charges will be lost.

## 13. Automated Retention Recommendations

Retention recommendations are intended to help businesses decide how to respond to customer churn risk and observed customer behavior.

Possible strategies include:

- **High-risk customers:** Review service issues, contact customers where appropriate, and investigate relevant retention offers.
- **Medium-risk customers:** Monitor customer engagement and review potential dissatisfaction.
- **Low-risk customers:** Continue regular service and customer engagement activities.

Recommendations should also consider customer tenure, contract type, service usage, charges, and support history where such information is available.

The project's retention recommendation output is stored in:

`reports/retention_recommendations.csv`

These recommendations are decision-support suggestions, not guarantees that a customer will remain with the business.

## 14. Business Value

The combined use of churn analysis, machine learning, customer segmentation, and risk categorization can support business teams in:

- Identifying customer groups that need attention.
- Understanding patterns associated with churn.
- Prioritizing customer retention efforts.
- Investigating service and pricing concerns.
- Making more informed decisions using historical data.

The effectiveness of these strategies should be measured through future retention outcomes and business feedback.

## 15. Flask Prediction Application

Flask is a Python web framework used to build web applications. In this project, Flask provides a web interface for the trained customer churn prediction model.

### 15.1 Application Workflow

The Flask application follows these steps:

1. Load the saved machine learning model.
2. Accept customer information through the web interface.
3. Prepare the supplied information for prediction.
4. Generate a churn prediction using the trained model.
5. Display the prediction result to the user.

The application uses the saved model:

`models/best_churn_model.joblib`

### 15.2 Running the Flask Application

Open a terminal in the project root directory and run:

```bash
python src/app.py
```

Open the following address in a browser:

`http://127.0.0.1:5000`

The address is available locally while the Flask development server is running.

### 15.3 Purpose

The Flask application demonstrates how a trained machine learning model can be integrated into a simple web-based prediction workflow.

The Flask development server is intended for local testing and demonstration, not direct production deployment.

## 16. Streamlit Analytics Dashboard

Streamlit is a Python framework for building interactive data applications.

The project includes a Streamlit dashboard to present customer churn analysis and project results in an accessible interface.

### 16.1 Dashboard Purpose

The dashboard helps users explore analytical outputs and understand important patterns in the customer dataset.

Depending on the available dashboard components, it can present:
- Customer churn summaries.
- Exploratory data analysis charts.
- Model evaluation results.
- Customer segment information.
- Churn risk analysis.
- Retention-related insights.

### 16.2 Running the Dashboard

Open a terminal in the project root directory and run:

```bash
python -m streamlit run src/streamlit_app.py
```

Streamlit will display the local URL in the terminal. Open that URL in a browser.

### 16.3 Business Use

A visual dashboard helps users understand analytical findings without having to inspect every CSV file or execute individual analysis scripts.

## 17. Automated Testing

Automated tests were created using Pytest to check important aspects of the trained model.

The test file is:

`tests/test_model.py`

### 17.1 Tests Implemented

The test suite contains four tests:

1. **Model file exists:** Checks whether the saved model file is available.
2. **Model loads successfully:** Checks whether the model can be loaded.
3. **Model supports prediction:** Checks whether the loaded model provides a prediction method.
4. **Model generates a prediction:** Checks whether the model can generate a prediction using a suitable input.

### 17.2 Running the Tests

Run the following command from the project root directory:

```bash
python -m pytest tests/test_model.py -v
```

The four implemented tests passed in the project's recorded test run.

### 17.3 Importance of Testing

Automated testing helps identify problems with model files and basic prediction functionality. Additional tests can be added for invalid inputs, missing values, and application behavior.

## 18. Project Folder Structure

The project is organized into folders according to their purpose.

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
|   |-- data_validation.py
|   |-- data_cleaning.py
|   |-- automated_eda.py
|   |-- feature_engineering.py
|   |-- prepare_ml_data.py
|   |-- train_model.py
|   |-- evaluate_model.py
|   |-- customer_segmentation.py
|   |-- predict_churn.py
|   |-- retention_recommendations.py
|   |-- save_visualizations.py
|
|-- models/
|   |-- best_churn_model.joblib
|
|-- reports/
|   |-- Visualization charts and analysis reports
|   |-- Customer_Churn_Full_Documentation.md
|
|-- tests/
|   |-- test_model.py
|
|-- README.md
|-- requirements.txt
```

This structure separates raw and processed data, source code, model artifacts, reports, notebooks, and tests to make the project easier to maintain.

## 19. Project Limitations

Although the project provides useful analytical insights, several limitations should be considered.

- The analysis is based on historical customer data and may not represent future customer behavior exactly.
- Model predictions are estimates and do not guarantee that a customer will leave.
- Model performance may change when customer behavior or business conditions change.
- Risk categories depend on the probability thresholds selected for this project.
- Monthly charges associated with high-risk customers do not represent confirmed revenue loss.
- Customer segments and retention recommendations require business interpretation and validation.
- The Flask application is intended for demonstration and local testing.
- The current automated tests cover basic model functionality and do not constitute complete application testing.

## 20. Future Improvements

The following improvements could make the project more useful:

- Evaluate additional machine learning algorithms and improve model performance.
- Validate prediction quality on newer customer data.
- Monitor changes in customer behavior and model performance.
- Add more automated tests for invalid inputs and application workflows.
- Improve the dashboard with additional business KPIs and filters.
- Refine customer segmentation using additional behavioral features.
- Evaluate retention recommendations using actual business outcomes.
- Add model version tracking and monitoring.
- Improve application security and deployment readiness before production use.

## 21. Conclusion

The Customer Intelligence & Predictive Retention Platform demonstrates how data analytics and machine learning can be applied to customer churn analysis.

The project includes data validation, data cleaning, feature engineering, Exploratory Data Analysis, machine learning model training and evaluation, customer segmentation, churn-risk analysis, and retention recommendations.

A Flask application demonstrates the integration of the trained model into a prediction interface, while a Streamlit dashboard presents analytical results visually. Automated tests provide basic checks for the saved model and its prediction functionality.

The project's charts and analytical outputs help explain customer behavior and identify areas for further investigation. These results can support customer retention planning, although predictions and recommendations should be validated against real business outcomes before operational use.

Overall, the project demonstrates a practical workflow combining data preparation, machine learning, visualization, application development, and testing in a customer analytics use case.

## 22. References

1. **IBM Telco Customer Churn Dataset** — The customer churn dataset used for this project.
2. **Pandas Documentation** — https://pandas.pydata.org/docs/
3. **Scikit-learn Documentation** — https://scikit-learn.org/stable/
4. **Matplotlib Documentation** — https://matplotlib.org/stable/
5. **Seaborn Documentation** — https://seaborn.pydata.org/
6. **Flask Documentation** — https://flask.palletsprojects.com/
7. **Streamlit Documentation** — https://docs.streamlit.io/
8. **Pytest Documentation** — https://docs.pytest.org/

## 23. Project Deliverables

The main deliverables of this project are:

- Original and processed customer datasets.
- Exploratory analysis notebook.
- Data validation, cleaning, and feature engineering scripts.
- Trained machine learning model.
- Model comparison and evaluation outputs.
- Customer segmentation and retention recommendation outputs.
- Visualization charts and analysis reports.
- Flask prediction application.
- Streamlit analytics dashboard.
- Automated model tests.
- Project README and full documentation.



