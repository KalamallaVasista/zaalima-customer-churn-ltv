# API Testing

## FastAPI Test Coverage

Automated API tests are implemented using Pytest and FastAPI TestClient.

Current tests cover:

- Root endpoint (`GET /`)
- Health endpoint (`GET /health`)
- Single-customer LTV prediction (`POST /predict/ltv`)

## Current Test Result

All implemented API tests are passing successfully.

- Tests executed: 3
- Tests passed: 3
- Tests failed: 0

Test command:

`pytest tests/test_api.py -v`

## Future Testing

Additional tests will be added during final integration for:

- Batch LTV prediction
- Churn prediction
- Retention priority scoring
- Invalid input handling