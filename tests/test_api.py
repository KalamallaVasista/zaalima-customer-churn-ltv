import sys
from pathlib import Path

from fastapi.testclient import TestClient


# ---------------------------------------------------------
# Project Setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


client = TestClient(app)


# ---------------------------------------------------------
# Shared Test Customer
# ---------------------------------------------------------

def sample_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.50,
        "TotalCharges": 1572.00,
    }


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

def test_root_endpoint():
    """Test the root endpoint."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

def test_health_endpoint():
    """Test API health and model loading."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["churn_model_loaded"] is True
    assert data["ltv_model_loaded"] is True


# ---------------------------------------------------------
# Single Churn Prediction
# ---------------------------------------------------------

def test_single_churn_prediction():
    """Test single-customer churn prediction."""

    customer = sample_customer()

    response = client.post(
        "/predict/churn",
        json=customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data
    assert "predicted_churn" in data
    assert "risk_segment" in data

    assert 0 <= data["churn_probability"] <= 100

    assert data["predicted_churn"] in [
        "Yes",
        "No",
    ]

    assert data["risk_segment"] in [
        "Low Risk",
        "Medium Risk",
        "High Risk",
    ]


# ---------------------------------------------------------
# Batch Churn Prediction
# ---------------------------------------------------------

def test_batch_churn_prediction():
    """Test batch churn prediction."""

    customers = [
        sample_customer(),
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 100.00,
            "TotalCharges": 6000.00,
        },
    ]

    response = client.post(
        "/predict/churn/batch",
        json=customers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_customers"] == 2

    predictions = data["predictions"]

    assert isinstance(predictions, list)
    assert len(predictions) == 2

    for prediction in predictions:

        assert "customer_number" in prediction
        assert "churn_probability" in prediction
        assert "predicted_churn" in prediction
        assert "risk_segment" in prediction

        assert (
            0
            <= prediction["churn_probability"]
            <= 100
        )

        assert prediction["predicted_churn"] in [
            "Yes",
            "No",
        ]

        assert prediction["risk_segment"] in [
            "Low Risk",
            "Medium Risk",
            "High Risk",
        ]


# ---------------------------------------------------------
# Single LTV Prediction
# ---------------------------------------------------------

def test_single_ltv_prediction():
    """Test single-customer LTV prediction."""

    customer = sample_customer()

    response = client.post(
        "/predict/ltv",
        json=customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_ltv" in data
    assert "ltv_segment" in data

    assert data["predicted_ltv"] >= 0

    assert data["ltv_segment"] in [
        "Low Value",
        "Medium Value",
        "High Value",
    ]


# ---------------------------------------------------------
# Batch LTV Prediction
# ---------------------------------------------------------

def test_batch_ltv_prediction():
    """Test batch customer LTV prediction."""

    customers = [
        sample_customer(),
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 100.00,
            "TotalCharges": 6000.00,
        },
    ]

    response = client.post(
        "/predict/ltv/batch",
        json=customers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_customers"] == 2

    predictions = data["predictions"]

    assert isinstance(predictions, list)
    assert len(predictions) == 2

    for prediction in predictions:

        assert "customer_number" in prediction
        assert "predicted_ltv" in prediction
        assert "ltv_segment" in prediction

        assert prediction["predicted_ltv"] >= 0

        assert prediction["ltv_segment"] in [
            "Low Value",
            "Medium Value",
            "High Value",
        ]


# ---------------------------------------------------------
# Invalid Input
# ---------------------------------------------------------

def test_invalid_customer_input():
    """Test incomplete customer data."""

    invalid_customer = {
        "gender": "Female",
        "tenure": 12,
        "MonthlyCharges": 70.00,
    }

    response = client.post(
        "/predict/ltv",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_invalid_churn_customer_input():
    """Test incomplete churn input."""

    invalid_customer = {
        "gender": "Male",
        "tenure": 5,
    }

    response = client.post(
        "/predict/churn",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_invalid_numeric_data_type():
    """Test invalid numeric datatype."""

    invalid_customer = sample_customer()

    invalid_customer["tenure"] = (
        "invalid-tenure"
    )

    response = client.post(
        "/predict/ltv",
        json=invalid_customer,
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Empty Batch Validation
# ---------------------------------------------------------

def test_empty_churn_batch():
    """Test empty churn batch."""

    response = client.post(
        "/predict/churn/batch",
        json=[],
    )

    assert response.status_code == 400


def test_empty_ltv_batch():
    """Test empty LTV batch."""

    response = client.post(
        "/predict/ltv/batch",
        json=[],
    )

    assert response.status_code == 400