# Customer Churn Prediction & Lifetime Value Engine

## Final Project Progress

### Project Objective

The objective of this project is to build a production-oriented analytics system that:

- Predicts customer churn.
- Estimates Customer Lifetime Value (LTV) for active customers.
- Explains important churn factors.
- Combines churn risk and customer value for retention prioritization.
- Stores analytics data in PostgreSQL.
- Provides predictions through FastAPI.
- Visualizes retention insights using Metabase.
- Supports deployment using Docker.

---

## 1. Dataset and Data Engineering

Dataset:

**Telco Customer Churn Dataset**

Dataset details:

- Total Customers: 7,043
- Original Columns: 21
- Churned Customers: 1,869
- Non-Churned Customers: 5,174
- Overall Churn Rate: 26.54%

Completed data-engineering activities:

- Dataset inspection
- Data-type analysis
- Missing-value investigation
- Duplicate-record validation
- Data cleaning
- Cleaned-data validation

### Data Quality Results

- Blank `TotalCharges` values identified: 11
- `TotalCharges` converted to numeric format
- Missing values after cleaning: 0
- Duplicate rows: 0

Cleaned dataset:

`data/processed/telco_customer_churn_clean.csv`

---

## 2. PostgreSQL Integration

PostgreSQL is used as the structured data-storage layer.

Completed:

- PostgreSQL database connection
- SQLAlchemy configuration
- Customer-data ingestion
- Database validation

Customer records loaded:

**7,043**

The final retention dataset was also loaded into the PostgreSQL table:

`retention_dashboard`

Retention dashboard records:

**5,174**

---

## 3. Exploratory Data Analysis

EDA was performed on the cleaned customer dataset.

### Key Findings

Contract churn rates:

- Month-to-month: 42.71%
- One year: 11.27%
- Two year: 2.83%

Other important observations:

- Fiber optic churn rate: 41.89%
- Electronic check churn rate: 45.29%
- No Tech Support churn rate: 41.64%
- No Online Security churn rate: 41.77%

Average tenure:

- Retained customers: 37.57 months
- Churned customers: 17.98 months

Average monthly charges:

- Retained customers: 61.27
- Churned customers: 74.44

EDA figures and summaries are stored in:

`reports/figures/eda/`

---

## 4. Feature Engineering

Seven additional customer-level features were created:

1. AvgMonthlySpend
2. TenureGroup
3. NumServices
4. IsMonthToMonth
5. HasInternet
6. AutoPayment
7. HasSecuritySupport

Dataset after feature engineering:

- Rows: 7,043
- Columns: 28

Feature dataset:

`data/processed/telco_customer_features.csv`

---

## 5. Churn Prediction Models

Three classification models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7374 | 0.5034 | 0.7888 | 0.6146 | 0.8424 |
| Random Forest | 0.7615 | 0.5446 | 0.6203 | 0.5800 | 0.8200 |
| XGBoost | 0.7580 | 0.5294 | 0.7941 | 0.6353 | 0.8442 |

### Final Churn Model

**XGBoost**

XGBoost was selected because it achieved the highest F1-score among the evaluated models.

The metrics above are based on the held-out test dataset.

The selected XGBoost pipeline was then retrained on the complete dataset for operational customer scoring.

---

## 6. SHAP Explainability

SHAP was used to explain the final XGBoost churn model.

Important model features include:

1. IsMonthToMonth
2. tenure
3. OnlineSecurity = No
4. MonthlyCharges
5. InternetService = Fiber optic
6. Contract = Month-to-month
7. TotalCharges
8. AvgMonthlySpend
9. PaymentMethod = Electronic check
10. TechSupport = No

SHAP outputs:

`reports/figures/churn_shap_bar.png`

`reports/figures/churn_shap_summary.png`

SHAP feature importance represents contribution magnitude. Observed churn-rate direction is supported separately through EDA.

---

## 7. Full Customer Churn Scoring

The final churn model was used to generate operational churn probabilities for all:

**7,043 customers**

### Risk Segments

- Low Risk: 3,217
- Medium Risk: 2,160
- High Risk: 1,666

Average churn probability:

**39.82%**

Risk rules:

- High Risk: probability >= 70%
- Medium Risk: probability >= 30% and < 70%
- Low Risk: probability < 30%

These thresholds are business rules used for segmentation.

The full-customer scores are operational scores generated after retraining on the complete historical dataset. The held-out test results remain the model evaluation results.

---

## 8. Customer Lifetime Value

LTV analysis was performed for active customers.

Active customers:

**5,174**

LTV-related fields:

- CurrentRevenue
- Projected12MonthRevenue
- ProjectedLTV

### LTV Model

Model:

**Random Forest Regressor**

Evaluation:

- MAE: 33.68
- RMSE: 55.16
- R²: 0.9996

### Important LTV Limitation

The source Telco dataset does not contain true observed future lifetime revenue.

Therefore, `ProjectedLTV` is a billing-derived proxy target.

The high R² is influenced by the close relationship between the proxy target and billing-related predictors. It should not be interpreted as 99.96% real-world prediction accuracy.

---

## 9. LTV Customer Segmentation

The 5,174 active customers were divided into:

- Low Value: 1,725
- Medium Value: 1,724
- High Value: 1,725

Output:

`data/processed/ltv_customer_segments.csv`

---

## 10. Churn and LTV Integration

Churn-risk results were integrated with LTV results for active customers.

Integration results:

- Total churn-risk records: 7,043
- Active LTV customers: 5,174
- Matched active customers: 5,174
- Active customers without churn-risk predictions: 0
- Integration coverage: 100%

The remaining 1,869 churn-risk records correspond to customers already marked as churned and therefore excluded from active-customer LTV analysis.

The 100% value represents integration coverage and not model accuracy.

---

## 11. Retention Priority

Retention priority was created by combining churn risk with customer LTV segment.

### Priority Distribution

- High Priority: 773
- Medium Priority: 1,920
- Low Priority: 2,481

### Average Churn Probability

- High Priority: 58.39%
- Medium Priority: 28.99%
- Low Priority: 18.40%

### Average Projected LTV

- High Priority: 5,168.49
- Medium Priority: 4,912.47
- Low Priority: 1,438.89

Overall active-customer results:

- Active Customers: 5,174
- Average Churn Probability: 28.30%
- Average Projected LTV: 3,285.09

Retention priority is a rule-based business decision layer and not a separate machine-learning model.

Dashboard dataset:

`data/processed/retention_dashboard_data.csv`

---

## 12. FastAPI Prediction Service

FastAPI provides model predictions through REST endpoints.

Available endpoints:

- `GET /`
- `GET /health`
- `POST /predict/churn`
- `POST /predict/churn/batch`
- `POST /predict/ltv`
- `POST /predict/ltv/batch`

The API supports:

- Single-customer churn prediction
- Batch churn prediction
- Churn probability
- Churn risk segmentation
- Single-customer LTV prediction
- Batch LTV prediction
- LTV segmentation
- Swagger API documentation

Swagger documentation:

`http://127.0.0.1:8000/docs`

---

## 13. Automated Testing

Automated API testing was completed using pytest.

Final result:

**11 tests passed**

Tests cover:

- Root endpoint
- Health endpoint
- Single churn prediction
- Batch churn prediction
- Single LTV prediction
- Batch LTV prediction
- Invalid customer input
- Invalid churn input
- Invalid numeric input
- Empty churn batch
- Empty LTV batch

The test suite completes successfully. Dependency deprecation warnings are present but do not cause test failures.

---

## 14. Metabase Dashboard

Metabase was connected to the PostgreSQL retention dataset.

Dashboard:

**Customer Churn & LTV Retention Dashboard**

The dashboard contains:

- Total Active Customers
- High Priority Customers
- Average Churn Probability
- Average Projected LTV
- Customers by Retention Priority
- Customers by Churn Risk Segment
- Customers by LTV Segment

The dashboard provides a business-level view of customer churn risk, customer value, and retention priorities.

---

## 15. Docker Deployment

The FastAPI application was containerized using Docker.

Docker image:

`zaalima-churn-ltv`

API container:

`zaalima-api`

API port:

`8000`

Metabase is also running through Docker on port:

`3000`

Final Docker API health validation:

- Status: healthy
- Churn model loaded: true
- LTV model loaded: true

---

## Final Project Status

The following project requirements have been completed:

- Data inspection and cleaning
- PostgreSQL integration
- Exploratory Data Analysis
- Feature engineering
- Logistic Regression
- Random Forest
- XGBoost
- Precision, Recall, F1 and ROC-AUC evaluation
- Final churn model selection
- SHAP explainability
- Full customer churn scoring
- Churn risk segmentation
- LTV regression
- LTV segmentation
- Churn and LTV integration
- Retention priority scoring
- FastAPI single prediction
- FastAPI batch prediction
- Automated API testing
- Metabase dashboard
- Docker containerization
- Technical documentation

The implementation is ready for final Git integration and project review.