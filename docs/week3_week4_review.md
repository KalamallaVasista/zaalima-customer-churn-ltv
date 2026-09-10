# Week 3 & Week 4 Project Review

## Week 3 - LTV and API Development

Week 3 focused on Customer Lifetime Value (LTV) modeling, churn-LTV integration, and API development.

### LTV Modeling
- Active customers used for LTV analysis: 5,174
- Random Forest Regressor used for Projected LTV estimation
- MAE: 33.68
- RMSE: 55.16
- R²: 0.9996
- Customers segmented into Low, Medium, and High Value groups

### Churn and LTV Integration
Churn risk and customer value were combined to generate retention priorities.

- High Priority: 773
- Medium Priority: 1,920
- Low Priority: 2,481

### FastAPI
REST API endpoints were developed for:
- Single churn prediction
- Batch churn prediction
- Single LTV prediction
- Batch LTV prediction
- Application health check

## Week 4 - Dashboard, Testing and Deployment

Week 4 focused on visualization, testing, containerization, and final project readiness.

### Metabase Dashboard
Metabase was connected to PostgreSQL to visualize:
- Active customers
- Churn probability
- Churn risk segments
- LTV segments
- Retention priorities

### Automated Testing
The FastAPI test suite contains 11 automated tests.

Final result:
- 11 tests passed

### Docker
The FastAPI application was containerized using Docker.
Metabase was also deployed using Docker.

### Documentation and Version Control
Project documentation was finalized and the completed implementation was pushed to the main GitHub branch.

## Important Limitation
Projected LTV is a billing-derived proxy because the source dataset does not contain observed future lifetime revenue. Therefore, the high LTV R² should not be interpreted as real-world prediction accuracy.