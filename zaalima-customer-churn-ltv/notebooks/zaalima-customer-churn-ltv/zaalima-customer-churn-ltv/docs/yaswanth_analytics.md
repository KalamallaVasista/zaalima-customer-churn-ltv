# Yaswanth - Churn Analytics and Machine Learning

## Responsibilities Completed

### 1. Exploratory Data Analysis
- Dataset inspection
- Missing-value investigation
- Duplicate checking
- Churn distribution
- Numerical variable analysis
- Categorical variable analysis
- Customer segmentation
- Correlation analysis

### 2. Feature Preparation
- Numerical feature identification
- Categorical feature identification
- One-hot encoding
- Numerical scaling
- Train/test split
- Prevention of test-data leakage

### 3. Churn Prediction Models
- Logistic Regression
- Random Forest
- XGBoost

### 4. Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## 5. Explainability

SHAP was used with the XGBoost model to identify the features that
contribute most strongly to churn predictions.

## 6. Customer Risk Segmentation

Customers were classified into:
- Low Risk
- Medium Risk
- High Risk

The risk segments are based on predicted churn probability.

## 7. Analytics Outputs

The following outputs were generated:

- `customer_risk_segments.csv`
- `model_comparison.csv`
- `churn_analytics_summary.csv`

These outputs can be consumed by the API/dashboard layer.
