import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


def test_single_ltv_prediction():
    """Test single-customer LTV prediction."""

    customer = {
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
        "TotalCharges": 1572.00
    }

    response = client.post("/predict/ltv", json=customer)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_ltv" in data
    assert "ltv_segment" in data

    assert data["predicted_ltv"] >= 0

    assert data["ltv_segment"] in [
        "Low Value",
        "Medium Value",
        "High Value"
    ]


def test_batch_ltv_prediction():
    """Test batch customer LTV prediction."""

    customers = [
        {
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
            "TotalCharges": 1572.00
        },
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
            "TotalCharges": 6000.00
        }
    ]

    response = client.post("/predict/ltv/batch", json=customers)

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)

    assert "total_customers" in data
    assert "predictions" in data

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
            "High Value"
        ]
def test_invalid_customer_input():
    """Test that incomplete customer data is rejected."""

    invalid_customer = {
        "gender": "Female",
        "tenure": 12,
        "MonthlyCharges": 70.00
    }

    response = client.post("/predict/ltv", json=invalid_customer)

    assert response.status_code == 422