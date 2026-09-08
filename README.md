# Customer Churn Prediction & Lifetime Value (LTV) Engine

A production-oriented Data Analytics and Machine Learning project developed for Zaalima Development.

The system predicts customer churn, estimates Customer Lifetime Value (LTV), explains churn drivers, identifies retention priorities, provides predictions through FastAPI, stores analytics data in PostgreSQL, visualizes business insights in Metabase, and supports Docker deployment.

## Team Members

- Vasi
- Yashwant

## Project Objective

The project aims to:

- Predict customers who are likely to churn.
- Estimate LTV for active customers.
- Explain the major factors influencing churn.
- Segment customers based on churn risk and LTV.
- Identify high-priority customers for retention.
- Store structured data in PostgreSQL.
- Serve predictions using FastAPI.
- Visualize retention insights using Metabase.
- Deploy the prediction API using Docker.

## Dataset

Telco Customer Churn Dataset

- Total customers: 7,043
- Original columns: 21
- Churned customers: 1,869
- Non-churned customers: 5,174
- Overall churn rate: 26.54%

## Project Workflow

1. Data inspection and cleaning
2. PostgreSQL data ingestion
3. Exploratory Data Analysis
4. Feature engineering
5. Churn model training and evaluation
6. SHAP explainability
7. LTV modeling and segmentation
8. Churn-risk and LTV integration
9. Retention-priority scoring
10. FastAPI prediction service
11. Metabase dashboard
12. Docker deployment
13. Automated API testing

## Exploratory Data Analysis

Important findings include:

- Month-to-month contract churn rate: 42.71%
- Fiber optic churn rate: 41.89%
- Electronic check churn rate: 45.29%
- Customers without Tech Support churn rate: 41.64%
- Customers without Online Security churn rate: 41.77%
- Average tenure of churned customers: 17.98 months
- Average tenure of retained customers: 37.57 months
- Average monthly charges of churned customers: 74.44
- Average monthly charges of retained customers: 61.27

EDA figures are available under:

`reports/figures/eda/`

## Feature Engineering

Seven additional customer-level features were created:

- AvgMonthlySpend
- TenureGroup
- NumServices
- IsMonthToMonth
- HasInternet
- AutoPayment
- HasSecuritySupport

Final feature dataset:

- Rows: 7,043
- Columns: 28

## Churn Prediction

Three classification models were evaluated:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7374 | 0.5034 | 0.7888 | 0.6146 | 0.8424 |
| Random Forest | 0.7615 | 0.5446 | 0.6203 | 0.5800 | 0.8200 |
| XGBoost | 0.7580 | 0.5294 | 0.7941 | 0.6353 | 0.8442 |

XGBoost was selected as the final churn model based on the highest F1-score.

The evaluation metrics above are based on the held-out test dataset.

## SHAP Explainability

SHAP was used to explain the final XGBoost model.

Major churn drivers include:

1. Month-to-month status
2. Tenure
3. No Online Security
4. Monthly Charges
5. Fiber optic Internet
6. Month-to-month contract
7. Total Charges
8. Average Monthly Spend
9. Electronic check payment
10. No Tech Support

SHAP figures are available under:

`reports/figures/`

## Customer Lifetime Value

LTV modeling is performed for 5,174 active customers.

Model:

Random Forest Regressor

Evaluation:

- MAE: 33.68
- RMSE: 55.16
- R²: 0.9996

### LTV Limitation

The source dataset does not contain true observed future lifetime revenue.

Therefore, `ProjectedLTV` is a billing-derived proxy target. The high R² should not be interpreted as 99.96% prediction accuracy.

## LTV Segmentation

Active customers are segmented into:

- Low Value
- Medium Value
- High Value

## Churn and LTV Integration

Churn-risk predictions were integrated with LTV results for all active customers.

- Active LTV customers: 5,174
- Active customers with churn-risk predictions: 5,174
- Integration coverage: 100%

The 100% value represents integration coverage, not model prediction accuracy.

## Retention Priority

Customers are prioritized using churn risk and LTV segment.

Final distribution:

- High Priority: 773
- Medium Priority: 1,920
- Low Priority: 2,481

Overall active-customer metrics:

- Average churn probability: 28.30%
- Average Projected LTV: 3,285.09

High-priority customers have:

- Average churn probability: 58.39%
- Average Projected LTV: 5,168.49

Retention priority is a rule-based business segmentation layer rather than a separate machine-learning model.

## PostgreSQL

PostgreSQL is used for structured data storage.

Tables include customer data and the final retention-dashboard dataset.

The dashboard table contains:

5,174 active-customer records.

## FastAPI

The prediction API supports both churn and LTV.

Endpoints:

- `GET /`
- `GET /health`
- `POST /predict/churn`
- `POST /predict/churn/batch`
- `POST /predict/ltv`
- `POST /predict/ltv/batch`

Swagger documentation:

`http://127.0.0.1:8000/docs`

Run locally:

```bash
uvicorn api.main:app --reload