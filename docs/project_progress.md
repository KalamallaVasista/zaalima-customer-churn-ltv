# Customer Churn Prediction & Lifetime Value Engine

## Project Progress

### Project Objective

The objective of this project is to build a production-oriented analytics system that:

- Predicts customer churn.
- Estimates Customer Lifetime Value (LTV).
- Helps identify high-value customers for retention.
- Stores structured customer data in PostgreSQL.
- Provides model predictions through FastAPI.
- Supports deployment using Docker.

---

## Dataset

Telco Customer Churn Dataset

- Total Customers: 7,043
- Original Columns: 21
- Churned Customers: 1,869
- Non-Churned Customers: 5,174

---

## 1. Data Engineering

Completed activities:

- Dataset inspection
- Data type analysis
- Missing-value investigation
- Duplicate-record validation
- Data cleaning
- Cleaned-data validation

### Data Quality Findings

- 11 blank values were identified in `TotalCharges`.
- `TotalCharges` was converted to numeric format.
- Missing values after cleaning: 0
- Duplicate rows: 0

Cleaned dataset:

`data/processed/telco_customer_churn_clean.csv`

---

## 2. PostgreSQL Integration

PostgreSQL is used as the structured data-storage layer.

Completed activities:

- PostgreSQL database connection
- SQLAlchemy configuration
- Customer-data ingestion
- Database validation

Total records loaded into PostgreSQL:

7,043

---

## 3. Feature Engineering

Seven additional customer-level features were created:

1. AvgMonthlySpend
2. TenureGroup
3. NumServices
4. IsMonthToMonth
5. HasInternet
6. AutoPayment
7. HasSecuritySupport

Dataset shape:

- Original: 7,043 rows × 21 columns
- After Feature Engineering: 7,043 rows × 28 columns

Feature dataset:

`data/processed/telco_customer_features.csv`

---

## 4. Customer Lifetime Value

LTV analysis is currently performed for active customers.

Active Customers:

5,174

LTV-related fields:

- CurrentRevenue
- Projected12MonthRevenue
- ProjectedLTV

### Regression Model

Model:

Random Forest Regressor

Evaluation:

- MAE: 33.68
- RMSE: 55.16
- R²: 0.9996

### Important Limitation

The source Telco dataset does not contain true observed future lifetime
revenue.

Therefore, `ProjectedLTV` is currently treated as a billing-derived proxy
for LTV. The high R² should not be interpreted as 99.96% real-world
prediction accuracy.

---

## 5. LTV Customer Segmentation

Active customers are divided into:

- Low Value
- Medium Value
- High Value

Output:

`data/processed/ltv_customer_segments.csv`

---

## 6. FastAPI

A FastAPI prediction service has been implemented.

Available endpoints:

- `GET /`
- `GET /health`
- `POST /predict/ltv`
- `POST /predict/ltv/batch`

The API supports:

- Single-customer LTV prediction
- Batch customer prediction
- LTV segmentation
- Swagger API documentation

---

## 7. Docker

The FastAPI application has been containerized using Docker.

Completed:

- Dockerfile creation
- Docker image build
- Docker container execution
- FastAPI execution inside Docker
- Prediction endpoint validation

---

## Current Integration Status

The following components are completed on the data-engineering/LTV branch:

- Data engineering
- PostgreSQL ingestion
- Feature engineering
- LTV preparation
- LTV regression
- LTV segmentation
- FastAPI
- Docker

---

## Remaining Project Work

- Finalize churn analytics review
- Integrate the final churn prediction model
- Generate churn probabilities through the API
- Combine churn risk with LTV
- Build customer retention priority scoring
- Complete Superset or Metabase dashboard
- Perform end-to-end testing
- Finalize technical documentation
- Integrate validated feature branches into `main`